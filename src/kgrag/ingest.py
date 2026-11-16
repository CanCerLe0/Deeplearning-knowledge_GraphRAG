"""离线校验 Markdown、按小节分片并写出可追溯的 JSONL。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import yaml

from .schema import Chunk, Document, SourceSpec


HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)\s*$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
LABEL_RE = re.compile(r"^:label:`([^`]+)`$")
PAGE_DIRECTIVE_RE = re.compile(r"^:(?:label|width|height):")
CODE_BUILD_RE = re.compile(
    r"^\s*(?:#@(?:tab|save)\b|%load_ext d2lbook\.tab|tab\.interact_select\()"
)
INLINE_BUILD_RE = re.compile(r"[ \t]+#@(?:tab|save)\b.*$")
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")


def load_sources(path: Path) -> list[SourceSpec]:
    """读取并校验来源清单；只接受显式列出的 Markdown。"""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("sources"), list):
        raise ValueError("来源清单必须包含 sources 列表")
    sources: list[SourceSpec] = []
    seen: set[str] = set()
    for number, item in enumerate(data["sources"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"第 {number} 条来源不是对象")
        try:
            source = SourceSpec(**item)
        except TypeError as error:
            raise ValueError(f"第 {number} 条来源字段不完整或包含未知字段") from error
        if source.source_id in seen:
            raise ValueError(f"重复的 source_id：{source.source_id}")
        seen.add(source.source_id)
        sources.append(source)
    return sources


def _read_document(source: SourceSpec, raw_dir: Path) -> Document:
    """按来源类型定位文件，并在解析前核对上游文件哈希。"""
    if source.kind == "github_markdown":
        assert source.repo is not None and source.sha256 is not None
        path = raw_dir / source.repo.split("/")[-1] / source.path
    else:
        # 本地自写文件的路径以项目根目录为基准。
        path = raw_dir.parent.parent / source.path
    raw = path.read_bytes()
    if source.sha256 is not None:
        actual = hashlib.sha256(raw).hexdigest()
        if actual != source.sha256:
            raise ValueError(f"SHA256 不匹配：{path}")
    return Document(source=source, text=raw.decode("utf-8"))


def _parse_sections(markdown: str) -> list[dict[str, Any]]:
    """识别真正的标题，保留正文原始行号并排除构建指令。"""
    sections: list[dict[str, Any]] = []
    stack: list[str] = []
    current: dict[str, Any] | None = None
    fence_character: str | None = None
    fence_length = 0

    for number, line in enumerate(markdown.splitlines(), start=1):
        fence = FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if fence_character is None:
                fence_character, fence_length = marker[0], len(marker)
            elif marker[0] == fence_character and len(marker) >= fence_length:
                fence_character, fence_length = None, 0
            if current is not None:
                current["lines"].append((number, line))
            continue
        if fence_character is not None:
            # D2L 的整行或行尾构建标记不属于检索正文。
            if current is not None and not CODE_BUILD_RE.match(line):
                current["lines"].append((number, INLINE_BUILD_RE.sub("", line)))
            continue

        heading = HEADING_RE.match(line)
        if heading:
            if current is not None:
                sections.append(current)
            level = len(heading.group(1))
            stack = stack[: level - 1]
            stack.append(heading.group(2).strip())
            current = {
                "path": tuple(stack),
                "anchor": None,
                "lines": [],
            }
            continue
        # 教材可能在第一个标题之前放执行环境代码；不作为知识片段。
        if current is None:
            continue
        label = LABEL_RE.match(line.strip())
        if label and current["anchor"] is None and not any(
            body.strip() for _, body in current["lines"]
        ):
            current["anchor"] = label.group(1)
            continue
        if PAGE_DIRECTIVE_RE.match(line.strip()):
            continue
        current["lines"].append((number, line))

    if current is not None:
        sections.append(current)
    return sections


def _blocks(section: dict[str, Any]) -> list[dict[str, Any]]:
    """按空行和围栏聚合段落、代码、公式；原子块不在内部切开。"""
    blocks: list[dict[str, Any]] = []
    prose: list[tuple[int, str]] = []
    special: list[tuple[int, str]] = []
    mode: str | None = None
    fence_character: str | None = None
    fence_length = 0

    def flush(lines: list[tuple[int, str]], atomic: bool) -> None:
        if not lines:
            return
        body = "\n".join(text for _, text in lines).strip()
        if body:
            blocks.append({
                "text": body,
                "line_start": lines[0][0],
                "line_end": lines[-1][0],
                "atomic": atomic,
            })
        lines.clear()

    for number, line in section["lines"]:
        fence = FENCE_RE.match(line)
        if mode == "fence":
            special.append((number, line))
            if fence and fence.group(1)[0] == fence_character and len(fence.group(1)) >= fence_length:
                flush(special, True)
                mode = None
            continue
        if mode == "math":
            special.append((number, line))
            if line.strip().endswith("$$"):
                flush(special, True)
                mode = None
            continue
        if fence:
            flush(prose, False)
            marker = fence.group(1)
            fence_character, fence_length = marker[0], len(marker)
            special.append((number, line))
            mode = "fence"
        elif line.strip().startswith("$$"):
            flush(prose, False)
            special.append((number, line))
            if len(line.strip()) > 2 and line.strip().endswith("$$"):
                flush(special, True)
            else:
                mode = "math"
        elif not line.strip():
            flush(prose, False)
        elif LIST_MARKER_RE.match(line):
            # 每个列表项独立成块，避免普通分片从条目中间开始。
            flush(prose, False)
            prose.append((number, line))
        else:
            prose.append((number, line))
    flush(prose, False)
    flush(special, True)
    return blocks


def _split_prose(
    block: dict[str, Any], max_units: int, overlap_units: int,
    measure: Callable[[str], int],
) -> list[dict[str, Any]]:
    """在指定字符或 token 预算内切长段落，并保留来源行号。"""
    text = block["text"]
    if block["atomic"] or measure(text) <= max_units:
        return [block]
    pieces: list[dict[str, Any]] = []
    start = 0
    while start < len(text):
        # 二分搜索预算允许的最大字符终点；实际单位由 measure 决定。
        low, high, end = start + 1, len(text), start
        while low <= high:
            middle = (low + high) // 2
            if measure(text[start:middle]) <= max_units:
                end = middle
                low = middle + 1
            else:
                high = middle - 1
        if end == start:
            raise ValueError("单个字符已超过分片预算")
        if end < len(text):
            tail = text[start + (end - start) // 2:end]
            boundaries = list(re.finditer(r"[。！？.!?；;]\s*|\s+", tail))
            if boundaries:
                end = start + (end - start) // 2 + boundaries[-1].end()
        part = text[start:end].strip()
        if part:
            left = text.find(part, start, end)
            pieces.append({
                "text": part,
                "line_start": block["line_start"] + text[:left].count("\n"),
                "line_end": block["line_start"] + text[: left + len(part)].count("\n"),
                "atomic": False,
            })
        if end == len(text):
            break
        # 从片段末尾向前找不超过重叠预算的最长后缀。
        if overlap_units:
            low, high, next_start = start + 1, end, end
            while low <= high:
                middle = (low + high) // 2
                if measure(text[middle:end]) <= overlap_units:
                    next_start = middle
                    high = middle - 1
                else:
                    low = middle + 1
            start = next_start
        else:
            start = end
    return pieces


def _chunk_document(
    document: Document, *, max_units: int, overlap_units: int,
    measure: Callable[[str], int],
) -> list[Chunk]:
    """先分小节，再合并短块，为每个片段生成稳定 ID 和来源链接。"""
    source = document.source
    document_id = hashlib.sha256(source.source_id.encode("utf-8")).hexdigest()[:16]
    chunks: list[Chunk] = []
    for section in _parse_sections(document.text):
        pieces = [
            piece
            for block in _blocks(section)
            for piece in _split_prose(block, max_units, overlap_units, measure)
        ]
        groups: list[list[dict[str, Any]]] = []
        group: list[dict[str, Any]] = []
        for piece in pieces:
            candidate = "\n\n".join(item["text"] for item in (*group, piece))
            if group and measure(candidate) > max_units:
                groups.append(group)
                group = []
            group.append(piece)
        if group:
            groups.append(group)

        for group_index, group in enumerate(groups):
            body = "\n\n".join(piece["text"] for piece in group)
            line_start, line_end = group[0]["line_start"], group[-1]["line_end"]
            section_name = " / ".join(section["path"])
            normalized = re.sub(r"\s+", " ", body).strip()
            identity = "\n".join((
                source.source_id, source.path, section["anchor"] or section_name,
                str(line_start), str(group_index), normalized,
            ))
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            url = None
            if source.kind == "github_markdown":
                url = (
                    f"https://github.com/{source.repo}/blob/{source.revision}/"
                    f"{source.path}#L{line_start}-L{line_end}"
                )
            chunks.append(Chunk(
                chunk_id=chunk_id,
                source_id=source.source_id,
                document_id=document_id,
                section=section_name,
                text=body,
                language=source.language,
                source_path=source.path,
                line_start=line_start,
                line_end=line_end,
                url=url,
            ))
    return chunks


def build_corpus(
    specs: list[SourceSpec], raw_dir: Path, out_dir: Path, *,
    max_chars: int = 1200, overlap_chars: int = 120,
    max_tokens: int | None = None, overlap_tokens: int = 48,
    token_counter: Callable[[str], int] | None = None,
) -> list[Chunk]:
    """完全离线地校验、分片并写出 chunks.jsonl。

    默认按字符计量。提供 max_tokens 与模型的 token_counter 后按实际
    token 数切分；代码和公式保持完整，可能超过目标。
    """
    if max_tokens is None:
        if token_counter is not None:
            raise ValueError("传入 token_counter 时必须同时设置 max_tokens")
        max_units, overlap_units, measure = max_chars, overlap_chars, len
    else:
        if token_counter is None:
            raise ValueError("按 token 分片需要提供模型的 token_counter")
        max_units, overlap_units, measure = max_tokens, overlap_tokens, token_counter
    if max_units < 4 or not 0 <= overlap_units < max_units // 2:
        raise ValueError("分片预算至少为 4，重叠预算须小于一半")
    raw_dir, out_dir = Path(raw_dir).resolve(), Path(out_dir).resolve()
    if len({source.source_id for source in specs}) != len(specs):
        raise ValueError("来源清单存在重复 source_id")

    # 先校验全部文件，再写输出；任一文件损坏都不会产出半份新索引。
    documents = [_read_document(source, raw_dir) for source in specs]
    chunks = [
        chunk
        for document in documents
        for chunk in _chunk_document(
            document, max_units=max_units, overlap_units=overlap_units, measure=measure
        )
    ]
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("生成了重复的 chunk_id")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=out_dir, prefix=".chunks-", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            for chunk in chunks:
                stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    # 构建清单只记录来源哈希与参数，不复制第三方正文。
    manifest = {
        "format_version": 1,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len(specs),
        "chunk_count": len(chunks),
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "max_tokens": max_tokens,
        "overlap_tokens": overlap_tokens if max_tokens is not None else None,
        "sources": [
            {
                "source_id": doc.source.source_id,
                "kind": doc.source.kind,
                "language": doc.source.language,
                "path": doc.source.path,
                "repo": doc.source.repo,
                "revision": doc.source.revision,
                "sha256": hashlib.sha256(doc.text.encode("utf-8")).hexdigest(),
            }
            for doc in documents
        ],
    }
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=out_dir, prefix=".manifest-", delete=False
    ) as stream:
        manifest_temporary = Path(stream.name)
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(manifest_temporary, out_dir / "build_manifest.json")
    os.replace(temporary, out_dir / "chunks.jsonl")
    return chunks
