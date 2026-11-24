"""开发题逐题评测：复用同一检索器、上下文预算与答案生成逻辑。"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from .answer import REFUSAL, SYSTEM_PROMPT, generate_answer
from .llm import ChatClient
from .retrieve import Retriever
from .schema import Answer

MODES = ("dense", "one_hop", "typed_path")
QUESTION_TYPES = ("single_concept", "multi_hop", "comparison", "unanswerable")


def _json_hash(value: object) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _run_config(
    questions_path: Path, retriever: Retriever, client: ChatClient | None,
    k: int, token_budget: int,
) -> dict:
    dense = retriever.dense
    vector_hash = hashlib.sha256()
    vector_hash.update(_json_hash({
        "shape": dense.vectors.shape, "dtype": str(dense.vectors.dtype),
        "chunks": [{
            "chunk_id": chunk.chunk_id, "source_id": chunk.source_id,
            "language": chunk.language, "url": chunk.url,
            "source_path": chunk.source_path,
        } for chunk in dense.chunks],
    }).encode("ascii"))
    vector_hash.update(dense.vectors.tobytes(order="C"))
    graph = retriever.graph
    graph_hash = None if graph is None else _json_hash({
        "chunk_ids": sorted(graph.chunks),
        "edges": sorted((asdict(edge) for edge in graph.edges), key=lambda item: (
            item["subject"], item["predicate"], item["object"], item["evidence_chunk_id"]
        )),
        "aliases": sorted(graph.aliases, key=lambda item: (
            item["canonical"], item["alias"]
        )),
        "reviews": sorted(graph.reviews, key=lambda item: json.dumps(item, sort_keys=True)),
        "rejected": graph.rejected,
    })
    embedder = dense.embedder
    return {
        "run_kind": "retrieval_only" if client is None else "answer",
        "questions_sha256": hashlib.sha256(Path(questions_path).read_bytes()).hexdigest(),
        "dense_sha256": vector_hash.hexdigest(),
        "graph_sha256": graph_hash,
        "answer_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "top_k": k,
        "token_budget": token_budget,
        "embedding_model_id": embedder.model_id,
        "embedding_revision": embedder.revision,
        "embedding_model_class": getattr(embedder, "model_class", None),
        "embedding_pooling_method": getattr(embedder, "pooling_method", None),
        "embedding_batch_size": getattr(embedder, "batch_size", None),
        "embedding_passage_max_length": getattr(embedder, "passage_max_length", None),
        "retrieval_config": asdict(retriever.config),
        "generation_model_id": getattr(client, "model", None),
        "temperature": getattr(client, "temperature", None),
        "max_output_tokens": getattr(client, "max_tokens", None),
    }


def _load_questions(path: Path, known_chunk_ids: set[str]) -> list[dict]:
    questions = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"第 {line_number} 行不是有效 JSON") from error
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise ValueError(f"第 {line_number} 行缺少题目 ID")
        if item["id"] in seen:
            raise ValueError(f"重复的题目 ID：{item['id']}")
        seen.add(item["id"])
        if not isinstance(item.get("question"), str) or not item["question"].strip():
            raise ValueError(f"第 {line_number} 行缺少问题")
        if item.get("type") not in QUESTION_TYPES:
            raise ValueError(f"第 {line_number} 行题型无效")
        if not isinstance(item.get("gold_answer"), str) or not item["gold_answer"].strip():
            raise ValueError(f"第 {line_number} 行缺少标准答案")
        groups = item.get("required_evidence_groups")
        if not isinstance(groups, list) or any(
            not isinstance(group, list) or not group or
            any(not isinstance(chunk_id, str) or chunk_id not in known_chunk_ids for chunk_id in group)
            for group in groups
        ):
            raise ValueError(f"第 {line_number} 行含无效证据组或未知片段 ID")
        if (item["type"] == "unanswerable") != (not groups):
            raise ValueError(f"第 {line_number} 行题型与证据组不一致")
        questions.append(item)
    if not questions:
        raise ValueError("开发题文件为空")
    return questions


def run_eval(
    questions_path: Path, modes: list[str], output_path: Path, *,
    retriever: Retriever, client: ChatClient | None, k: int = 5, token_budget: int = 3000,
) -> None:
    """每题每模式生成一条 JSONL；失败时保留原有结果文件。"""
    if not modes or any(mode not in MODES for mode in modes) or len(set(modes)) != len(modes):
        raise ValueError("评测模式必须是互不重复的 dense、one_hop 或 typed_path")
    if k < 1 or token_budget < 1:
        raise ValueError("top-k 和 token 预算必须为正整数")
    questions = _load_questions(
        questions_path, {chunk.chunk_id for chunk in retriever.dense.chunks}
    )
    config = _run_config(questions_path, retriever, client, k, token_budget)
    rows = []
    for item in questions:
        for mode in modes:
            start = perf_counter()
            evidence = retriever.retrieve(item["question"], mode, k, token_budget)
            retrieved_at = perf_counter()
            if client is None:
                answer = None
                usage = {}
                model_version = None
            elif evidence:
                answer = generate_answer(item["question"], evidence, client)
                usage = dict(getattr(client, "last_usage", {}))
                model_version = getattr(client, "last_model", None)
            else:
                answer = Answer(REFUSAL, [])
                usage = {}
                model_version = None
            finished_at = perf_counter()
            rows.append({
                "question_id": item["id"],
                "question": item["question"],
                "question_type": item.get("type"),
                "mode": mode,
                "retrieved_chunk_ids": [entry.chunk.chunk_id for entry in evidence],
                "evidence_languages": sorted({entry.chunk.language for entry in evidence}),
                "evidence": [{
                    "chunk_id": entry.chunk.chunk_id,
                    "language": entry.chunk.language,
                    "source_id": entry.chunk.source_id,
                    "source_path": entry.chunk.source_path,
                    "section": entry.chunk.section,
                    "line_start": entry.chunk.line_start,
                    "line_end": entry.chunk.line_end,
                    "url": entry.chunk.url,
                    **({"text": entry.chunk.text} if client is not None else {}),
                    "score": entry.score,
                    "path": list(entry.path),
                    "path_relations": list(entry.path_relations),
                    "path_evidence_chunk_ids": list(entry.path_evidence_chunk_ids),
                    "path_directions": list(entry.path_directions),
                } for entry in evidence],
                "answer": answer.text if answer is not None else None,
                "citation_chunk_ids": answer.citation_chunk_ids if answer is not None else [],
                "latency_ms": round((finished_at - start) * 1000, 3),
                "retrieval_latency_ms": round((retrieved_at - start) * 1000, 3),
                "generation_latency_ms": round((finished_at - retrieved_at) * 1000, 3),
                "usage": usage,
                "generation_model_version": model_version,
                "run_config": config,
            })
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent,
        prefix=f".{output_path.name}-", delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
