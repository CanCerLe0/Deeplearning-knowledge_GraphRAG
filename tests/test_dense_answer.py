"""无网络、无真实模型的双语向量检索与证据回答测试。"""

from __future__ import annotations

import json
import io
import os
import tempfile
import types
import unittest
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

from kgrag.answer import generate_answer
from kgrag.embed import DenseIndex, FlagEmbedder
from kgrag.llm import DeepSeekChatClient
from kgrag.schema import Chunk


class FakeEmbedder:
    model_id = "test/multilingual"
    revision = "fixture-v1"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = {
            "注意力和查询如何关联？": [1.0, 0.0],
            "Attention uses a query to score keys.": [1.0, 0.0],
            "注意力机制聚合值。": [0.0, 1.0],
            "Transformer has an encoder and decoder.": [0.4, 0.6],
        }
        return np.asarray([vectors[text] for text in texts], dtype=np.float32)


def chunk(chunk_id: str, text: str, language: str, url: str | None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, source_id=f"source-{chunk_id}", document_id=f"doc-{chunk_id}",
        section="Attention", text=text, language=language, source_path=f"{chunk_id}.md",
        line_start=1, line_end=2, url=url,
    )


class DenseAnswerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.embedder = FakeEmbedder()
        self.chunks = [
            chunk("zh", "注意力机制聚合值。", "zh", None),
            chunk("en", "Attention uses a query to score keys.", "en", "https://example.org/en#L1-L2"),
            chunk("other", "Transformer has an encoder and decoder.", "en", None),
        ]

    def test_chinese_query_ranks_english_evidence_first_after_reload(self) -> None:
        DenseIndex.build(self.chunks, self.embedder, self.out)
        index = DenseIndex.load(self.out, self.embedder)
        results = index.search("注意力和查询如何关联？", 2)
        self.assertEqual([item.chunk.chunk_id for item in results], ["en", "other"])
        self.assertEqual(results[0].chunk.language, "en")
        self.assertEqual(results[0].chunk.url, "https://example.org/en#L1-L2")
        self.assertEqual(results[0].path, ())
        self.assertAlmostEqual(results[0].score, 1.0)
        manifest = json.loads((self.out / "dense_manifest.json").read_text())
        self.assertEqual(manifest["embedding_model_id"], "test/multilingual")
        self.assertEqual(manifest["embedding_revision"], "fixture-v1")
        self.assertEqual(manifest["normalization"], "l2")

    def test_search_does_not_emit_numeric_warnings_for_finite_vectors(self) -> None:
        rng = np.random.default_rng(0)
        vectors = rng.normal(size=(555, 1024)).astype(np.float32)
        query = vectors[0].copy()

        class WideEmbedder:
            def encode(self, texts: list[str]) -> np.ndarray:
                return query[None, :]

        index = DenseIndex([self.chunks[0]] * len(vectors), vectors, WideEmbedder())
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always", RuntimeWarning)
            results = index.search("attention", 1)
        self.assertEqual(len(results), 1)
        self.assertTrue(np.isfinite(results[0].score))
        self.assertEqual(observed, [])

    def test_rejects_different_embedding_model_on_load(self) -> None:
        DenseIndex.build(self.chunks, self.embedder, self.out)

        class OtherEmbedder(FakeEmbedder):
            model_id = "test/other"

        with self.assertRaisesRegex(ValueError, "embedding"):
            DenseIndex.load(self.out, OtherEmbedder())

    def test_rejects_mixed_index_files_even_when_dimensions_match(self) -> None:
        DenseIndex.build(self.chunks, self.embedder, self.out)
        chunks_path = self.out / "dense_chunks.jsonl"
        chunks_path.write_text(chunks_path.read_text().replace("source-en", "source-other"))
        with self.assertRaisesRegex(ValueError, "索引"):
            DenseIndex.load(self.out, self.embedder)

    def test_pinned_bge_snapshot_uses_explicit_model_class(self) -> None:
        called: dict = {}
        downloaded: dict = {}

        class FakeModel:
            def encode(self, texts: list[str]) -> dict:
                return {"dense_vecs": np.asarray([[1.0, 2.0] for _ in texts])}

        class FakeAutoModel:
            @staticmethod
            def from_finetuned(path: str, **kwargs: object) -> FakeModel:
                called.update(path=path, **kwargs)
                if kwargs.get("model_class") != "encoder-only-m3":
                    raise AssertionError("snapshot path does not identify the model class")
                return FakeModel()

        modules = {
            "FlagEmbedding": types.SimpleNamespace(FlagAutoModel=FakeAutoModel),
            "huggingface_hub": types.SimpleNamespace(
                snapshot_download=lambda **kwargs: (
                    downloaded.update(kwargs) or "/cache/snapshots/" + "a" * 40
                )
            ),
        }
        with patch.dict("sys.modules", modules):
            embedder = FlagEmbedder("BAAI/bge-m3", revision="a" * 40, device="cpu")
            with self.assertRaisesRegex(ValueError, "长度"):
                FlagEmbedder("BAAI/bge-m3", passage_max_length=0)
        self.assertEqual(embedder.revision, "a" * 40)
        self.assertEqual(called["model_class"], "encoder-only-m3")
        self.assertEqual(called["passage_max_length"], 2048)
        self.assertEqual(called["batch_size"], 8)
        self.assertIn("onnx/*", downloaded["ignore_patterns"])
        self.assertEqual(embedder.encode(["中文"]).shape, (1, 2))

    def test_empty_evidence_refuses_without_calling_client(self) -> None:
        class NoCallClient:
            def complete_json(self, messages: list[dict]) -> dict:
                raise AssertionError("no evidence must not call the model")

        answer = generate_answer("未覆盖的问题？", [], NoCallClient())
        self.assertEqual(answer.text, "证据不足，无法根据现有资料回答。")
        self.assertEqual(answer.citation_chunk_ids, [])

    def test_filters_citations_outside_supplied_evidence(self) -> None:
        index = DenseIndex.build(self.chunks, self.embedder, self.out)
        evidence = index.search("注意力和查询如何关联？", 1)

        class FakeClient:
            def complete_json(self, messages: list[dict]) -> dict:
                self.messages = messages
                return {
                    "answer": "查询与键计算相关性。[en]",
                    "citation_chunk_ids": ["en", "invented", "en"],
                }

        client = FakeClient()
        answer = generate_answer("注意力和查询如何关联？", evidence, client)
        self.assertEqual(answer.text, "查询与键计算相关性。[en]")
        self.assertEqual(answer.citation_chunk_ids, ["en"])
        prompt = json.dumps(client.messages, ensure_ascii=False)
        self.assertIn("中文", prompt)
        self.assertIn("en", prompt)
        self.assertIn("Attention uses a query to score keys.", prompt)
        self.assertIn("language", prompt)

    def test_rejects_invalid_inline_citation(self) -> None:
        index = DenseIndex.build(self.chunks, self.embedder, self.out)
        evidence = index.search("注意力和查询如何关联？", 1)

        class FakeClient:
            def complete_json(self, messages: list[dict]) -> dict:
                return {
                    "answer": "查询与键计算相关性。[en] 无依据的结论。[invented]",
                    "citation_chunk_ids": ["en", "invented"],
                }

        answer = generate_answer("注意力和查询如何关联？", evidence, FakeClient())
        self.assertEqual(answer.citation_chunk_ids, [])
        self.assertIn("证据不足", answer.text)

    def test_rejects_uncited_second_claim(self) -> None:
        index = DenseIndex.build(self.chunks, self.embedder, self.out)
        evidence = index.search("注意力和查询如何关联？", 1)

        class FakeClient:
            def complete_json(self, messages: list[dict]) -> dict:
                return {
                    "answer": "查询与键计算相关性。[en] 这还能保证任意模型收敛。",
                    "citation_chunk_ids": ["en"],
                }

        answer = generate_answer("注意力和查询如何关联？", evidence, FakeClient())
        self.assertEqual(answer.citation_chunk_ids, [])
        self.assertIn("证据不足", answer.text)

    def test_cli_build_and_ask_with_fake_adapters(self) -> None:
        from kgrag.cli import main

        chunks_path = self.out / "chunks.jsonl"
        chunks_path.write_text(
            "".join(json.dumps(asdict(item), ensure_ascii=False) + "\n" for item in self.chunks),
            encoding="utf-8",
        )
        index_dir = self.out / "dense"

        class FakeClient:
            def __init__(self, **kwargs: object) -> None:
                self.model = "fake-chat"
                self.last_model = "fake-chat-v1"
                self.temperature = 0.0
                self.max_tokens = 128
                self.last_usage: dict = {}

            def complete_json(self, messages: list[dict]) -> dict:
                return {"answer": "查询与键计算相关性。[en]", "citation_chunk_ids": ["en"]}

        output = StringIO()
        with patch("kgrag.cli.FlagEmbedder", return_value=self.embedder), patch(
            "kgrag.cli.DeepSeekChatClient", FakeClient
        ), redirect_stdout(output):
            self.assertEqual(main(["build-index", "--chunks", str(chunks_path), "--out-dir", str(index_dir)]), 0)
            output.seek(0)
            output.truncate(0)
            self.assertEqual(main(["ask", "注意力和查询如何关联？", "--index-dir", str(index_dir), "--top-k", "1"]), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["answer"]["citation_chunk_ids"], ["en"])
        self.assertEqual(result["evidence"][0]["language"], "en")
        self.assertEqual(result["evidence"][0]["url"], "https://example.org/en#L1-L2")
        self.assertEqual(result["run_config"]["generation_model_version"], "fake-chat-v1")

    def test_cli_no_evidence_refuses_without_api_client(self) -> None:
        from kgrag.cli import main

        DenseIndex.build(self.chunks, self.embedder, self.out)
        output = StringIO()

        def no_client(**kwargs: object) -> object:
            raise AssertionError("no evidence must not create the API client")

        with patch("kgrag.cli.FlagEmbedder", return_value=self.embedder), patch(
            "kgrag.cli.DeepSeekChatClient", no_client
        ), redirect_stdout(output):
            self.assertEqual(main([
                "ask", "注意力和查询如何关联？", "--index-dir", str(self.out), "--top-k", "0"
            ]), 0)
        self.assertIn("证据不足", json.loads(output.getvalue())["answer"]["text"])

    def test_deepseek_client_sends_json_mode_and_records_usage(self) -> None:
        sent = {}
        response = {
            "choices": [{"finish_reason": "stop", "message": {"content": '{"answer":"你好"}'}}],
            "model": "test-chat-v1",
            "usage": {"total_tokens": 17},
        }

        def fake_urlopen(request: object, timeout: int) -> io.BytesIO:
            sent["url"] = request.full_url
            sent["authorization"] = request.get_header("Authorization")
            sent["body"] = json.loads(request.data)
            return io.BytesIO(json.dumps(response).encode())

        client = DeepSeekChatClient(api_key="fixture-key", model="test-chat", temperature=0.0)
        with patch("kgrag.llm.urlopen", fake_urlopen):
            result = client.complete_json([{"role": "user", "content": "输出 JSON"}])
        self.assertEqual(result, {"answer": "你好"})
        self.assertEqual(sent["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(sent["authorization"], "Bearer fixture-key")
        self.assertEqual(sent["body"]["model"], "test-chat")
        self.assertEqual(sent["body"]["response_format"], {"type": "json_object"})
        self.assertEqual(sent["body"]["thinking"], {"type": "disabled"})
        self.assertEqual(client.last_usage, {"total_tokens": 17})
        self.assertEqual(client.last_model, "test-chat-v1")

    def test_current_deepseek_model_is_the_default(self) -> None:
        client = DeepSeekChatClient(api_key="fixture-key")
        self.assertEqual(client.model, "deepseek-flash")

    def test_deepseek_client_reads_ignored_local_env_file(self) -> None:
        (self.out / ".env").write_text("# Local secret\nDEEPSEEK_API_KEY=fixture-local-key\n")
        original_directory = Path.cwd()
        try:
            os.chdir(self.out)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
                client = DeepSeekChatClient()
        finally:
            os.chdir(original_directory)
        self.assertEqual(client.api_key, "fixture-local-key")

    def test_deepseek_client_prefers_environment_over_local_env_file(self) -> None:
        (self.out / ".env").write_text("DEEPSEEK_API_KEY=fixture-local-key\n")
        original_directory = Path.cwd()
        try:
            os.chdir(self.out)
            with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "fixture-env-key"}):
                client = DeepSeekChatClient()
        finally:
            os.chdir(original_directory)
        self.assertEqual(client.api_key, "fixture-env-key")


if __name__ == "__main__":
    unittest.main()
