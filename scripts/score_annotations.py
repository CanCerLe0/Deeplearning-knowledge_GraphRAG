"""生成不显示检索模式的人工评阅表，并汇总已填写的评分。"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from kgrag.evaluate import summarize_retrieval


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_blind_template(
    questions: list[dict], rows: list[dict], *, seed: int = 20261124,
) -> tuple[list[dict], list[dict]]:
    by_id = {item["id"]: item for item in questions}
    if len(by_id) != len(questions):
        raise ValueError("题目 ID 重复")
    seen = set()
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    template, key = [], []
    for number, row in enumerate(shuffled, 1):
        question_id = row["question_id"]
        identity = (question_id, row["mode"])
        if question_id not in by_id or identity in seen or row.get("answer") is None:
            raise ValueError("盲评需要非重复且已生成答案的有效题目结果")
        seen.add(identity)
        question = by_id[question_id]
        blind_id = f"blind-{number:04d}"
        template.append({
            "blind_id": blind_id,
            "question_id": question_id,
            "question_type": question["type"],
            "question": question["question"],
            "gold_answer": question["gold_answer"],
            "answer": row["answer"],
            "citation_chunk_ids": row["citation_chunk_ids"],
            "evidence": [{key: entry[key] for key in (
                "chunk_id", "language", "url", "text"
            ) if key in entry} for entry in row["evidence"]],
            "answer_rating": None,  # 0=错误，1=部分正确，2=正确
            "citation_supported": None,
            "refusal_correct": None,
            "failure_category": None,
            "notes": "",
        })
        key.append({"blind_id": blind_id, "question_id": question_id, "mode": row["mode"]})
    return template, key


def score_annotations(annotations: list[dict], key: list[dict]) -> dict:
    by_blind = {item["blind_id"]: item for item in key}
    if len(by_blind) != len(key) or len(annotations) != len(key):
        raise ValueError("盲评条数或映射 ID 不一致")
    grouped = defaultdict(list)
    seen = set()
    for item in annotations:
        blind_id = item["blind_id"]
        if blind_id not in by_blind or blind_id in seen:
            raise ValueError("盲评 ID 重复或未知")
        seen.add(blind_id)
        if type(item.get("answer_rating")) is not int or item["answer_rating"] not in (0, 1, 2):
            raise ValueError(f"{blind_id} 尚未填写有效答案评分")
        if item["citation_chunk_ids"] and type(item.get("citation_supported")) is not bool:
            raise ValueError(f"{blind_id} 尚未核查引用")
        if item["question_type"] == "unanswerable" and type(item.get("refusal_correct")) is not bool:
            raise ValueError(f"{blind_id} 尚未核查拒答")
        grouped[by_blind[blind_id]["mode"]].append(item)

    result = {}
    for mode, items in grouped.items():
        unanswerable = [item for item in items if item["question_type"] == "unanswerable"]
        cited = [item for item in items if item["citation_chunk_ids"]]
        result[mode] = {
            "question_count": len(items),
            "answer_correct_rate": sum(item["answer_rating"] == 2 for item in items) / len(items),
            "answer_partial_rate": sum(item["answer_rating"] == 1 for item in items) / len(items),
            "citation_support_rate": (
                sum(item["citation_supported"] for item in cited) / len(cited) if cited else None
            ),
            "refusal_accuracy": (
                sum(item["refusal_correct"] for item in unanswerable) / len(unanswerable)
                if unanswerable else None
            ),
            "failure_cases": [{"question_id": item["question_id"],
                               "category": item["failure_category"], "notes": item["notes"]}
                              for item in items if item.get("failure_category")],
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--questions", type=Path, required=True)
    template.add_argument("--results", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--key", type=Path, required=True)
    template.add_argument("--seed", type=int, default=20261124)
    score = commands.add_parser("score")
    score.add_argument("--annotations", type=Path, required=True)
    score.add_argument("--key", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    retrieval = commands.add_parser("retrieval")
    retrieval.add_argument("--questions", type=Path, required=True)
    retrieval.add_argument("--results", type=Path, required=True)
    retrieval.add_argument("--output", type=Path, required=True)
    retrieval.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.command == "template":
        rows, key = make_blind_template(
            _read_jsonl(args.questions), _read_jsonl(args.results), seed=args.seed
        )
        _write_jsonl(args.output, rows)
        _write_jsonl(args.key, key)
    elif args.command == "score":
        summary = score_annotations(_read_jsonl(args.annotations), _read_jsonl(args.key))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        summary = summarize_retrieval(
            _read_jsonl(args.questions), _read_jsonl(args.results), args.top_k
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
