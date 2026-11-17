"""用伪下载器验证固定版本下载，无需访问网络。"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from kgrag.fetch import fetch_sources
from kgrag.schema import SourceSpec


class FetchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "data/raw"

    def github_source(self, repo: str, revision: str, path: str, body: bytes) -> SourceSpec:
        return SourceSpec(
            source_id=f"{repo}:{path}", kind="github_markdown", path=path,
            title="Test chapter", language="zh" if repo.endswith("zh") else "en",
            license_note="self-written test fixture", repo=repo, revision=revision,
            sha256=hashlib.sha256(body).hexdigest(),
        )

    def test_downloads_only_listed_files_at_each_pinned_revision(self) -> None:
        zh_body = b"# Chinese\nAttention.\n"
        en_body = b"# English\nTransformer.\n"
        zh = self.github_source("d2l-ai/d2l-zh", "a" * 40, "chapter/attention.md", zh_body)
        en = self.github_source("d2l-ai/d2l-en", "b" * 40, "chapter/transformer.md", en_body)
        local = SourceSpec(
            source_id="sample", kind="local_markdown", path="data/sample/extra.md",
            title="Sample", language="en", license_note="self-written",
        )
        expected = {
            f"https://raw.githubusercontent.com/d2l-ai/d2l-zh/{'a' * 40}/chapter/attention.md": zh_body,
            f"https://raw.githubusercontent.com/d2l-ai/d2l-en/{'b' * 40}/chapter/transformer.md": en_body,
        }
        requested: list[str] = []

        def download(url: str) -> bytes:
            requested.append(url)
            return expected[url]

        paths = fetch_sources([local, zh, en], self.raw, download)
        self.assertEqual(requested, list(expected))
        self.assertEqual(paths, [
            self.raw / "d2l-zh/chapter/attention.md",
            self.raw / "d2l-en/chapter/transformer.md",
        ])
        self.assertEqual([path.read_bytes() for path in paths], [zh_body, en_body])
        self.assertFalse((self.raw / "d2l-zh/chapter/attention_origin.md").exists())

    def test_verified_files_are_reused_without_downloader(self) -> None:
        body = b"# Cached\nValid content.\n"
        source = self.github_source("d2l-ai/d2l-en", "c" * 40, "chapter/cached.md", body)
        path = self.raw / "d2l-en/chapter/cached.md"
        path.parent.mkdir(parents=True)
        path.write_bytes(body)

        def offline(_: str) -> bytes:
            self.fail("verified local files must not trigger a download")

        self.assertEqual(fetch_sources([source], self.raw, offline), [path])
        self.assertEqual(path.read_bytes(), body)

    def test_bad_download_is_rejected_without_writing_target(self) -> None:
        source = self.github_source("d2l-ai/d2l-zh", "d" * 40, "chapter/bad.md", b"expected")
        target = self.raw / "d2l-zh/chapter/bad.md"
        with self.assertRaisesRegex(ValueError, "SHA256"):
            fetch_sources([source], self.raw, lambda _: b"wrong")
        self.assertFalse(target.exists())

    def test_corrupt_cached_file_aborts_without_using_downloader(self) -> None:
        source = self.github_source("d2l-ai/d2l-en", "e" * 40, "chapter/cached.md", b"expected")
        target = self.raw / "d2l-en/chapter/cached.md"
        target.parent.mkdir(parents=True)
        target.write_bytes(b"wrong")

        def download(_: str) -> bytes:
            self.fail("corrupt cached files must stop the build")

        with self.assertRaisesRegex(ValueError, "SHA256"):
            fetch_sources([source], self.raw, download)
        self.assertEqual(target.read_bytes(), b"wrong")

    def test_cli_fetch_reuses_verified_file_offline(self) -> None:
        body = b"# Cached\nValid content.\n"
        source = self.github_source("d2l-ai/d2l-en", "f" * 40, "chapter/cached.md", body)
        target = self.raw / "d2l-en/chapter/cached.md"
        target.parent.mkdir(parents=True)
        target.write_bytes(body)
        config = self.root / "sources.yaml"
        config.write_text(
            "sources:\n"
            "  - source_id: cached\n"
            "    kind: github_markdown\n"
            "    path: chapter/cached.md\n"
            "    title: Cached\n"
            "    language: en\n"
            "    license_note: self-written\n"
            "    repo: d2l-ai/d2l-en\n"
            f"    revision: {'f' * 40}\n"
            f"    sha256: {source.sha256}\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, "-m", "kgrag.cli", "fetch", "--sources", str(config),
             "--raw-dir", str(self.raw)],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(target.read_bytes(), body)


if __name__ == "__main__":
    unittest.main()
