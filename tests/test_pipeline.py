"""Task 5 的离线端到端验收，所有教材文字均为自写。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from kgrag.answer import REFUSAL
from kgrag.embed import DenseIndex
from kgrag.fetch import fetch_sources
from kgrag.graph import GraphIndex
from kgrag.ingest import build_corpus
from kgrag.retrieve import RetrievalConfig, Retriever
from kgrag.schema import Edge, SourceSpec


class TopicEmbedder:
    model_id = "fixture/topics"
    revision = "v1"

    def encode(self, texts):
        vectors = []
        for text in texts:
            if "新增局部文档" in text or "新知识" in text:
                vectors.append([0.0, 0.0, 1.0])
            elif "Multi-head" in text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([1.0, 0.0, 0.0])
        return np.asarray(vectors, dtype=np.float32)


class AnswerClient:
    last_model = "fixture/answer-v1"
    last_usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}

    def complete_json(self, messages):
        payload = json.loads(messages[-1]["content"])
        english = [item for item in payload["evidence"] if item["language"] == "en"]
        if not english:
            return {"answer": REFUSAL, "citation_chunk_ids": []}
        return {"answer": f"多头注意力用于 Transformer。[{english[0]['chunk_id']}]",
                "citation_chunk_ids": [english[0]["chunk_id"]]}


def _fixture_corpus(tmp_path: Path):
    zh = "# 多头注意力\n\n多头注意力是注意力机制的一种。\n"
    en = "# Multi-head attention\n\nMulti-head attention is used in Transformer.\n"
    revision = "a" * 40
    specs = [
        SourceSpec("zh-fixture", "github_markdown", "chapter/attention.md", "注意力", "zh",
                   "fixture", "example/zh", revision, hashlib.sha256(zh.encode()).hexdigest()),
        SourceSpec("en-fixture", "github_markdown", "chapter/multihead.md", "Attention", "en",
                   "fixture", "example/en", revision, hashlib.sha256(en.encode()).hexdigest()),
    ]
    downloads = {"attention.md": zh.encode(), "multihead.md": en.encode()}
    raw_dir = tmp_path / "data" / "raw"

    def downloader(url):
        return downloads[url.rsplit("/", 1)[-1]]

    fetch_sources(specs, raw_dir, downloader)
    chunks = build_corpus(specs, raw_dir, tmp_path / "data" / "index")
    return specs, raw_dir, chunks


def test_bilingual_fetch_ingest_retrieve_answer_and_evaluate(tmp_path: Path):
    from kgrag.evaluate import run_eval

    specs, raw_dir, chunks = _fixture_corpus(tmp_path)
    fetch_sources(specs, raw_dir, lambda url: pytest.fail("已校验文件不应重新下载"))
    en_id = next(item.chunk_id for item in chunks if item.language == "en")
    zh_id = next(item.chunk_id for item in chunks if item.language == "zh")
    embedder = TopicEmbedder()
    dense_dir = tmp_path / "dense"
    DenseIndex.build(chunks, embedder, dense_dir)
    dense = DenseIndex.load(dense_dir, embedder)
    graph_dir = tmp_path / "graph"
    aliases = [{"canonical": "多头注意力", "alias": "Multi-head attention",
                "canonical_chunk_id": zh_id, "alias_chunk_id": en_id,
                "verified_by": "fixture-review"}]
    GraphIndex(chunks, [Edge("Multi-head attention", "used_in", "Transformer", en_id, 0.9)],
               aliases, {}).save(graph_dir, manifest={"format_version": 1})
    retriever = Retriever(dense, GraphIndex.load(graph_dir), RetrievalConfig(seed_k=1))
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({
        "id": "fixture-1", "question": "多头注意力在 Transformer 中有什么用途？",
        "type": "multi_hop", "gold_answer": "用于 Transformer。",
        "required_evidence_groups": [[en_id]],
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "results.jsonl"

    run_eval(questions, ["dense", "one_hop", "typed_path"], output,
             retriever=retriever, client=AnswerClient(), k=2, token_budget=1000)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["mode"] for row in rows] == ["dense", "one_hop", "typed_path"]
    assert all(row["question_id"] == "fixture-1" and row["latency_ms"] >= 0 for row in rows)
    assert all(row["usage"]["total_tokens"] == 20 for row in rows)
    for row in rows:
        assert row["citation_chunk_ids"] == [en_id]
        assert row["answer"].startswith("多头注意力")
        assert "github.com/example/en/blob/" in row["evidence"][0]["url"] or any(
            "github.com/example/en/blob/" in item["url"] for item in row["evidence"]
        )
        assert "en" in row["evidence_languages"]
    assert rows[-1]["evidence"][0]["path_evidence_chunk_ids"] == [en_id]


def test_offline_rebuild_with_local_markdown_preserves_old_ids_and_links(tmp_path: Path):
    specs, raw_dir, old_chunks = _fixture_corpus(tmp_path)
    old_provenance = {item.chunk_id: item.url for item in old_chunks}
    local_path = tmp_path / "data" / "sample" / "new.md"
    local_path.parent.mkdir(parents=True)
    local_path.write_text("# 新增局部文档\n\n新知识用于补充本地知识库。\n", encoding="utf-8")
    specs.append(SourceSpec("local-new", "local_markdown", "data/sample/new.md", "新增知识",
                            "zh", "original fixture"))
    fetch_sources(specs, raw_dir, lambda url: pytest.fail("离线重建不应下载"))
    chunks = build_corpus(specs, raw_dir, tmp_path / "rebuilt")
    assert {item.chunk_id: item.url for item in chunks if item.chunk_id in old_provenance} == old_provenance
    dense = DenseIndex.build(chunks, TopicEmbedder(), tmp_path / "rebuilt" / "dense")
    evidence = Retriever(dense).retrieve("新增局部文档中的新知识是什么？", "dense", 1, 1000)
    assert evidence[0].chunk.source_id == "local-new"
    assert evidence[0].chunk.url is None
    assert evidence[0].chunk.source_path == "data/sample/new.md"
    assert evidence[0].chunk.line_start >= 1
    assert evidence[0].chunk.line_end >= evidence[0].chunk.line_start


def test_no_evidence_refuses_without_calling_generation(tmp_path: Path):
    from kgrag.evaluate import run_eval

    _, _, chunks = _fixture_corpus(tmp_path)
    dense = DenseIndex.build(chunks, TopicEmbedder(), tmp_path / "dense")
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "unknown", "question": "语料未涉及的内容？",
                                     "type": "unanswerable", "gold_answer": REFUSAL,
                                     "required_evidence_groups": []}, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    output = tmp_path / "result.jsonl"

    run_eval(questions, ["dense"], output, retriever=Retriever(dense),
             client=AnswerClient(), token_budget=1)

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["answer"] == REFUSAL
    assert row["citation_chunk_ids"] == []
    assert row["evidence"] == []
    assert row["usage"] == {}


def test_cli_eval_runs_all_modes_from_saved_indexes(tmp_path: Path):
    from kgrag.cli import main

    _, _, chunks = _fixture_corpus(tmp_path)
    en_id = next(item.chunk_id for item in chunks if item.language == "en")
    zh_id = next(item.chunk_id for item in chunks if item.language == "zh")
    dense_dir = tmp_path / "dense"
    graph_dir = tmp_path / "graph"
    DenseIndex.build(chunks, TopicEmbedder(), dense_dir)
    aliases = [{"canonical": "多头注意力", "alias": "Multi-head attention",
                "canonical_chunk_id": zh_id, "alias_chunk_id": en_id,
                "verified_by": "fixture-review"}]
    GraphIndex(chunks, [Edge("Multi-head attention", "used_in", "Transformer", en_id, 0.9)],
               aliases, {}).save(graph_dir, manifest={"format_version": 1})
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps({"id": "q1", "question": "多头注意力用于哪里？",
                                     "type": "single_concept", "gold_answer": "用于 Transformer。",
                                     "required_evidence_groups": [[en_id]]}, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    output = tmp_path / "eval.jsonl"

    with patch("kgrag.cli.FlagEmbedder", return_value=TopicEmbedder()), patch(
        "kgrag.cli.DeepSeekChatClient", return_value=AnswerClient()
    ):
        assert main(["eval", "--questions", str(questions), "--output", str(output),
                     "--index-dir", str(dense_dir), "--graph-dir", str(graph_dir),
                     "--top-k", "2", "--token-budget", "1000"]) == 0

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert {row["mode"] for row in rows} == {"dense", "one_hop", "typed_path"}
    assert all(row["run_config"]["top_k"] == 2 for row in rows)

    retrieval_output = tmp_path / "retrieval_only.jsonl"
    with patch("kgrag.cli.FlagEmbedder", return_value=TopicEmbedder()), patch(
        "kgrag.cli.DeepSeekChatClient", side_effect=AssertionError("不应创建生成客户端")
    ):
        assert main(["eval", "--questions", str(questions), "--output", str(retrieval_output),
                     "--index-dir", str(dense_dir), "--graph-dir", str(graph_dir),
                     "--retrieval-only"]) == 0
    retrieval_rows = [json.loads(line) for line in retrieval_output.read_text().splitlines()]
    assert len(retrieval_rows) == 3
    assert all(row["answer"] is None for row in retrieval_rows)


def test_retrieval_only_records_reproducible_fingerprints_without_generation(tmp_path: Path):
    from kgrag.evaluate import run_eval

    _, _, chunks = _fixture_corpus(tmp_path)
    dense = DenseIndex.build(chunks, TopicEmbedder(), tmp_path / "dense")
    en_id = next(item.chunk_id for item in chunks if item.language == "en")
    graph = GraphIndex(chunks, [Edge("Multi-head attention", "used_in", "Transformer",
                                     en_id, 0.9)], [], {})
    question = {"id": "q1", "question": "多头注意力用于哪里？", "type": "single_concept",
                "gold_answer": "用于 Transformer。", "required_evidence_groups": [[en_id]]}
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    output = tmp_path / "retrieval.jsonl"

    run_eval(questions, ["dense", "typed_path"], output,
             retriever=Retriever(dense, graph), client=None, k=2)

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["answer"] is None and row["usage"] == {} for row in rows)
    assert all("text" not in entry for row in rows for entry in row["evidence"])
    assert all(row["run_config"]["questions_sha256"] == hashlib.sha256(
        questions.read_bytes()).hexdigest() for row in rows)
    assert all(len(row["run_config"]["dense_sha256"]) == 64 for row in rows)
    assert all(len(row["run_config"]["graph_sha256"]) == 64 for row in rows)
    assert all(len(row["run_config"]["answer_prompt_sha256"]) == 64 for row in rows)
    assert rows[1]["run_config"]["embedding_batch_size"] is None
    previous_graph_hash = rows[1]["run_config"]["graph_sha256"]
    graph.edges.append(Edge("Transformer", "used_in", "多头注意力", en_id, 0.8))
    run_eval(questions, ["dense"], output, retriever=Retriever(dense, graph), client=None)
    changed = json.loads(output.read_text(encoding="utf-8"))
    assert changed["run_config"]["graph_sha256"] != previous_graph_hash


@pytest.mark.parametrize("change", [
    {"type": "unknown"},
    {"gold_answer": ""},
    {"required_evidence_groups": []},
])
def test_eval_rejects_malformed_development_labels(tmp_path: Path, change: dict):
    from kgrag.evaluate import run_eval

    _, _, chunks = _fixture_corpus(tmp_path)
    dense = DenseIndex.build(chunks, TopicEmbedder(), tmp_path / "dense")
    question = {"id": "q1", "question": "有何用途？", "type": "single_concept",
                "gold_answer": "答案。", "required_evidence_groups": [[chunks[0].chunk_id]]}
    question.update(change)
    questions = tmp_path / "questions.jsonl"
    questions.write_text(json.dumps(question, ensure_ascii=False) + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        run_eval(questions, ["dense"], tmp_path / "out.jsonl",
                 retriever=Retriever(dense), client=AnswerClient())
