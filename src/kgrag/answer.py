"""仅依据给定原文片段生成中文回答并校验引用。"""

from __future__ import annotations

import json

from .llm import ChatClient
from .schema import Answer, Evidence


REFUSAL = "证据不足，无法根据现有资料回答。"
SYSTEM_PROMPT = (
    "你是深度学习知识库助手。仅根据提供的原文证据用中文回答；"
    "证据不足时明确回答证据不足。输出 JSON 对象，包含 answer 字符串和 "
    "citation_chunk_ids 数组；数组只能引用提供的 chunk_id。"
    "answer 文本不写引用标记，引用由 citation_chunk_ids 单独表示。"
)


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
    citations = list(dict.fromkeys(
        item for item in raw_citations if isinstance(item, str) and item in supplied
    ))
    if not text.strip() or not citations:
        return Answer(REFUSAL, [])
    return Answer(text.strip(), citations)
