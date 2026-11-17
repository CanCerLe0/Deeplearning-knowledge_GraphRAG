"""按来源清单下载固定版本的 Markdown，并校验本地文件。"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable
from urllib.request import urlopen

from .schema import SourceSpec


def download_url(url: str) -> bytes:
    """CLI 使用的网络下载器；测试可向 fetch_sources 注入替代实现。"""
    with urlopen(url, timeout=30) as response:
        return response.read()


def fetch_sources(
    specs: list[SourceSpec], raw_dir: Path, downloader: Callable[[str], bytes]
) -> list[Path]:
    """只获取清单中的 GitHub 文件；校验通过的文件可离线复用。"""
    raw_dir = Path(raw_dir)
    paths: list[Path] = []
    for source in specs:
        if source.kind != "github_markdown":
            continue
        assert source.repo is not None and source.revision is not None
        assert source.sha256 is not None
        target = raw_dir / source.repo.split("/")[-1] / source.path
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != source.sha256:
                raise ValueError(f"SHA256 不匹配：{target}")
            paths.append(target)
            continue

        url = (
            f"https://raw.githubusercontent.com/{source.repo}/"
            f"{source.revision}/{source.path}"
        )
        data = downloader(url)
        if hashlib.sha256(data).hexdigest() != source.sha256:
            raise ValueError(f"下载文件 SHA256 不匹配：{url}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".fetch-", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(data)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, target)
        paths.append(target)
    return paths
