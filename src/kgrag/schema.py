"""来源、片段、检索证据和回答的数据结构。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal


@dataclass(frozen=True)
class SourceSpec:
    """一条可离线定位的 Markdown 来源。"""

    source_id: str
    kind: Literal["github_markdown", "local_markdown"]
    path: str
    title: str
    language: Literal["zh", "en"]
    license_note: str
    repo: str | None = None
    revision: str | None = None
    sha256: str | None = None

    def __post_init__(self) -> None:
        # 来源路径必须相对根目录，避免清单越界读取其他文件。
        path = PurePosixPath(self.path)
        if not self.source_id or not self.title or not self.license_note:
            raise ValueError("来源 ID、标题和许可备注不能为空")
        if path.is_absolute() or ".." in path.parts or path.suffix != ".md":
            raise ValueError(f"无效的 Markdown 相对路径：{self.path}")
        if path.name.endswith("_origin.md"):
            raise ValueError(f"排除翻译原稿副本 _origin.md：{self.path}")
        if self.language not in ("zh", "en"):
            raise ValueError(f"不支持的语言：{self.language}")
        if self.kind == "github_markdown":
            if not self.repo or not re.fullmatch(r"[^/]+/[^/]+", self.repo):
                raise ValueError("GitHub 来源必须填写 owner/repo")
            if not self.revision or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
                raise ValueError("GitHub 来源必须填写完整的 40 位提交 SHA")
            if not self.sha256 or not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
                raise ValueError("GitHub 来源必须填写 64 位文件 SHA256")
        elif self.kind != "local_markdown":
            raise ValueError(f"不支持的来源类型：{self.kind}")


@dataclass(frozen=True)
class Document:
    """已读取并校验的单篇 Markdown。"""

    source: SourceSpec
    text: str


@dataclass(frozen=True)
class Chunk:
    """可直接供后续索引读取的片段与来源信息。"""

    chunk_id: str
    source_id: str
    document_id: str
    section: str
    text: str
    language: Literal["zh", "en"]
    source_path: str
    line_start: int
    line_end: int
    url: str | None


@dataclass(frozen=True)
class Edge:
    """经本地校验的概念关系；证据始终是一个原文片段。"""

    subject: str
    predicate: Literal["prerequisite_of", "component_of", "contrasts_with", "used_in"]
    object: str
    evidence_chunk_id: str
    confidence: float


@dataclass(frozen=True)
class Evidence:
    """检索到的原文片段及其得分和图路径。"""

    chunk: Chunk
    score: float
    path: tuple[str, ...] = ()
    path_relations: tuple[str, ...] = ()
    path_evidence_chunk_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Answer:
    """中文回答及实际证据片段 ID。"""

    text: str
    citation_chunk_ids: list[str]
