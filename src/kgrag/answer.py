"""仅依据给定原文片段生成中文回答并校验引用。"""

from __future__ import annotations

import json
import re

from .llm import ChatClient
from .schema import Answer, Evidence


REFUSAL = "证据不足，无法根据现有资料回答。"
SYSTEM_PROMPT = (
    "你是深度学习知识库助手。仅根据提供的原文证据用中文回答；"
    "证据不足时明确回答证据不足。输出 JSON 对象，包含 answer 字符串和 "
    "citation_chunk_ids 数组；数组只能引用提供的 chunk_id。"
    "answer 中每项事实后都要写 [chunk_id] 引用标记，引用由 citation_chunk_ids 同时列出。"
)
INLINE_CITATION_RE = re.compile(r"\[([A-Za-z0-9_-]+)\]")
CLAIM_RE = re.compile(r"[^。！？；\n]+(?:[。！？；]+(?:\s*\[[A-Za-z0-9_-]+\])*)?")


def generate_answer(question: str, evidence: list[Evidence], client: ChatClient) -> Answer:
    """过滤不在本次证据中的 ID；没有有效引用时拒答。"""
    if not evidence:
        return Answer(REFUSAL, [])
    supplied = {item.chunk.chunk_id for item in evidence}
    passages = [
        {
            "chunk_id": item.chunk.chunk_id,
            "language": item.chunk.language,
            "section": item.chunk.section,
            "text": item.chunk.text,
            "url": item.chunk.url,
        }
        for item in evidence
    ]
    result = client.complete_json([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(
            {"question": question, "evidence": passages}, ensure_ascii=False
        )},
    ])
    text = result.get("answer")
    raw_citations = result.get("citation_chunk_ids")
    if not isinstance(text, str) or not isinstance(raw_citations, list):
        return Answer(REFUSAL, [])
    listed = list(dict.fromkeys(
        item for item in raw_citations if isinstance(item, str) and item in supplied
    ))
    inline = list(dict.fromkeys(INLINE_CITATION_RE.findall(text)))
    claims = [part.strip() for part in CLAIM_RE.findall(text) if part.strip()]
    if (not text.strip() or not inline or any(item not in supplied for item in inline)
            or set(inline) != set(listed)
            or any(not INLINE_CITATION_RE.search(part) for part in claims)):
        return Answer(REFUSAL, [])
    return Answer(text.strip(), inline)
