"""用自写 Markdown 验证离线入库，不读取教材或外部服务。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kgrag.ingest import build_corpus, load_sources
from kgrag.schema import SourceSpec


class IngestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data/raw"
        self.out = self.root / "data/index"

    def github_source(self, text: str) -> SourceSpec:
        path = self.raw / "d2l-zh/chapter/demo.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return SourceSpec(
            source_id="d2l-zh:chapter/demo.md",
            kind="github_markdown",
            path="chapter/demo.md",
            title="示例",
            language="zh",
            license_note="自写测试文本",
            repo="example/d2l-zh",
            revision="a" * 40,
            sha256=hashlib.sha256(text.encode()).hexdigest(),
        )

    def test_load_sources_reads_both_kinds_and_rejects_duplicate_ids(self) -> None:
        config = self.root / "sources.yaml"
        config.write_text(
            "sources:\n"
            "  - source_id: local-one\n"
            "    kind: local_markdown\n"
            "    path: data/sample/demo.md\n"
            "    title: 本地文档\n"
            "    language: zh\n"
            "    license_note: 自写\n"
            "  - source_id: remote-one\n"
            "    kind: github_markdown\n"
            "    path: chapter/demo.md\n"
            "    title: Remote\n"
            "    language: en\n"
            "    license_note: test\n"
            "    repo: example/d2l-en\n"
            f"    revision: {'b' * 40}\n"
            f"    sha256: {'c' * 64}\n",
            encoding="utf-8",
        )
        specs = load_sources(config)
        self.assertEqual([spec.language for spec in specs], ["zh", "en"])
        self.assertEqual(specs[1].revision, "b" * 40)
        with config.open("a", encoding="utf-8") as stream:
            stream.write(
                "  - source_id: local-one\n"
                "    kind: local_markdown\n"
                "    path: data/sample/other.md\n"
                "    title: 冲突\n"
                "    language: zh\n"
                "    license_note: 自写\n"
            )
        with self.assertRaisesRegex(ValueError, "重复"):
            load_sources(config)

    def test_build_preserves_code_math_and_provenance_across_rebuilds(self) -> None:
        label = ":label:" + chr(96) + "sec_demo" + chr(96)
        text = (
            "~~~python\n# 正文前的伪标题\n~~~\n"
            f"# 示例\n{label}\n\n一段说明。\n\n"
            "~~~python\n# 代码里的伪标题\n#@tab pytorch\n"
            "def run():  #@save\n    return 1\n~~~\n\n"
            "$$\nx^2 + y^2\n$$\n\n## 子节\n更多说明。\n"
        )
        source = self.github_source(text)
        first = build_corpus([source], self.raw, self.out, max_chars=100, overlap_chars=10)
        second = build_corpus([source], self.raw, self.out, max_chars=100, overlap_chars=10)
        self.assertEqual(first, second)
        self.assertTrue(first)
        self.assertTrue(any("def run():" in chunk.text for chunk in first))
        self.assertTrue(any("$$\nx^2 + y^2\n$$" in chunk.text for chunk in first))
        self.assertFalse(any("正文前" in chunk.text for chunk in first))
        self.assertFalse(any("伪标题" in chunk.section for chunk in first))
        self.assertFalse(any("#@tab" in chunk.text or "#@save" in chunk.text for chunk in first))
        self.assertTrue(all(chunk.language == "zh" for chunk in first))
        self.assertTrue(all(chunk.url.startswith("https://github.com/example/d2l-zh/blob/" + "a" * 40) for chunk in first))
        self.assertTrue(all(chunk.line_start <= chunk.line_end for chunk in first))
        records = [json.loads(line) for line in (self.out / "chunks.jsonl").read_text().splitlines()]
        self.assertEqual([item["chunk_id"] for item in records], [chunk.chunk_id for chunk in first])

        # 增加本地文档时，旧片段 ID 与固定版本 URL 必须保持不变。
        local_path = self.root / "data/sample/extra.md"
        local_path.parent.mkdir(parents=True)
        local_path.write_text("# Extra\nA new document.\n", encoding="utf-8")
        local = SourceSpec(
            source_id="local-extra", kind="local_markdown",
            path="data/sample/extra.md", title="Extra",
            language="en", license_note="自写",
        )
        expanded = build_corpus([source, local], self.raw, self.out, max_chars=100, overlap_chars=10)
        self.assertEqual(expanded[:len(first)], first)
        self.assertEqual(expanded[-1].language, "en")
        self.assertIsNone(expanded[-1].url)

    def test_hash_mismatch_aborts_without_writing_index(self) -> None:
        source = self.github_source("# Safe\nOriginal.\n")
        path = self.raw / "d2l-zh/chapter/demo.md"
        path.write_text("# Safe\nChanged.\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            build_corpus([source], self.raw, self.out)
        self.assertFalse((self.out / "chunks.jsonl").exists())

    def test_rejects_origin_translation_copy(self) -> None:
        with self.assertRaisesRegex(ValueError, "_origin"):
            SourceSpec(
                source_id="copy", kind="local_markdown",
                path="data/sample/chapter_origin.md", title="Copy",
                language="zh", license_note="自写",
            )

    def test_build_records_source_hash_and_chunk_configuration(self) -> None:
        source = self.github_source("# Example\nA short sentence.\n")
        chunks = build_corpus([source], self.raw, self.out, max_chars=80, overlap_chars=8)
        manifest = json.loads((self.out / "build_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["max_chars"], 80)
        self.assertEqual(manifest["overlap_chars"], 8)
        self.assertEqual(manifest["chunk_count"], len(chunks))
        self.assertEqual(manifest["sources"][0]["sha256"], source.sha256)
        self.assertEqual(manifest["sources"][0]["revision"], "a" * 40)

    def test_long_prose_has_overlap_and_stable_ids(self) -> None:
        source = self.github_source("# Long\n" + "这是第一句。第二句继续解释。" * 18)
        chunks = build_corpus([source], self.raw, self.out, max_chars=80, overlap_chars=8)
        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(len(chunk.text) <= 80 for chunk in chunks))
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))
        self.assertTrue(any(left.text[-8:] in right.text for left, right in zip(chunks, chunks[1:])))

    def test_model_token_counter_limits_prose_chunks(self) -> None:
        source = self.github_source(
            "# Words\n" + " ".join(f"word{i}" for i in range(45)) + "\n"
        )
        count_words = lambda text: len(text.split())
        chunks = build_corpus(
            [source], self.raw, self.out,
            max_tokens=12, overlap_tokens=2, token_counter=count_words,
        )
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(count_words(chunk.text) <= 12 for chunk in chunks))
        manifest = json.loads((self.out / "build_manifest.json").read_text())
        self.assertEqual(manifest["max_tokens"], 12)
        self.assertEqual(manifest["overlap_tokens"], 2)

    def test_list_items_form_separate_chunk_boundaries(self) -> None:
        source = self.github_source(
            "# List\n"
            "- alpha " + "one " * 8 + "\n"
            "- beta " + "two " * 8 + "\n"
        )
        chunks = build_corpus([source], self.raw, self.out, max_chars=55, overlap_chars=5)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].text.startswith("- alpha"))
        self.assertTrue(chunks[1].text.startswith("- beta"))

    def test_cli_ingest_reads_sources_without_network(self) -> None:
        local_path = self.root / "data/sample/attention.md"
        local_path.parent.mkdir(parents=True)
        local_path.write_text("# Attention\nLocal example.\n", encoding="utf-8")
        config = self.root / "sources.yaml"
        config.write_text(
            "sources:\n"
            "  - source_id: attention-local\n"
            "    kind: local_markdown\n"
            "    path: data/sample/attention.md\n"
            "    title: Attention\n"
            "    language: en\n"
            "    license_note: self-written\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "kgrag.cli", "ingest",
                "--sources", str(config),
                "--raw-dir", str(self.raw),
                "--out-dir", str(self.out),
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        records = [json.loads(line) for line in (self.out / "chunks.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_id"], "attention-local")


if __name__ == "__main__":
    unittest.main()
