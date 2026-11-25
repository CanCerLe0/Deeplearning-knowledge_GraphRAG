"""证据组指标应按论点计数，而不是按片段或语种计数。"""

import pytest
import json
from collections import Counter
from pathlib import Path

from kgrag.evaluate import complete_evidence_hit, recall_at_k, summarize_retrieval
from scripts.score_annotations import make_blind_template, score_annotations


def test_partial_hit_counts_one_of_two_required_arguments():
    groups = [{"zh-a", "en-a"}, {"zh-b"}]
    retrieved = ["en-a", "zh-a", "irrelevant", "zh-b"]
    assert recall_at_k(groups, retrieved, 3) == 0.5
    assert complete_evidence_hit(groups, retrieved, 3) is False
    assert recall_at_k(groups, retrieved, 5) == 1.0
    assert complete_evidence_hit(groups, retrieved, 5) is True


def test_unanswerable_does_not_enter_recall_average():
    questions = [
        {"id": "a", "type": "multi_hop", "required_evidence_groups": [["zh-a"], ["en-b"]],
         "gold_evidence_languages": ["zh", "en"]},
        {"id": "b", "type": "unanswerable", "required_evidence_groups": []},
    ]
    rows = [
        {"question_id": "a", "mode": "dense", "retrieved_chunk_ids": ["zh-a"],
         "evidence_languages": ["zh"], "latency_ms": 10, "usage": {}},
        {"question_id": "b", "mode": "dense", "retrieved_chunk_ids": [],
         "evidence_languages": [], "latency_ms": 20, "usage": {}},
    ]
    summary = summarize_retrieval(questions, rows, k=5)
    assert summary["dense"]["overall"]["recall_at_5"] == 0.5
    assert summary["dense"]["overall"]["group_micro_recall_at_5"] == 0.5
    assert summary["dense"]["overall"]["recall_question_count"] == 1
    assert summary["dense"]["overall"]["unanswerable_count"] == 1
    assert summary["dense"]["by_question_type"]["multi_hop"]["complete_evidence_rate"] == 0.0
    assert summary["dense"]["failure_cases"] == [{"question_id": "a", "missing_group_indexes": [1]}]
    assert summary["dense"]["overall"]["mean_latency_ms"] == 15.0
    assert summary["dense"]["by_gold_language"]["en+zh"]["question_count"] == 1


def test_empty_evidence_groups_have_no_recall_value():
    assert recall_at_k([], ["x"], 5) is None
    assert complete_evidence_hit([], ["x"], 5) is None
    with pytest.raises(ValueError):
        recall_at_k([set()], ["x"], 5)


def test_retrieval_summary_rejects_incomplete_mode():
    questions = [{"id": "q1", "type": "single_concept", "required_evidence_groups": [["a"]]},
                 {"id": "q2", "type": "single_concept", "required_evidence_groups": [["b"]]}]
    rows = [{"question_id": "q1", "mode": "dense", "retrieved_chunk_ids": ["a"],
             "evidence_languages": ["zh"], "latency_ms": 1}]
    with pytest.raises(ValueError, match="缺少"):
        summarize_retrieval(questions, rows)


def test_blind_template_hides_mode_and_scores_checked_answers():
    questions = [
        {"id": "q1", "question": "问题一？", "type": "single_concept", "gold_answer": "答案一。",
         "required_evidence_groups": [["a"]]},
        {"id": "q2", "question": "问题二？", "type": "unanswerable",
         "gold_answer": "证据不足。", "required_evidence_groups": []},
    ]
    rows = [
        {"question_id": "q1", "mode": "dense", "answer": "答案一。", "citation_chunk_ids": ["a"],
         "evidence": [{"chunk_id": "a", "text": "原文", "url": "https://example.org/a", "language": "zh"}]},
        {"question_id": "q2", "mode": "dense", "answer": "证据不足。", "citation_chunk_ids": [],
         "evidence": []},
    ]
    template, key = make_blind_template(questions, rows, seed=7)
    assert len(template) == 2
    assert all("mode" not in item and item["answer_rating"] is None for item in template)
    assert all("mode" in item for item in key)
    for item in template:
        item["answer_rating"] = 2
        if item["citation_chunk_ids"]:
            item["citation_supported"] = True
        if item["question_type"] == "unanswerable":
            item["refusal_correct"] = True
    scored = score_annotations(template, key)
    assert scored["dense"]["answer_correct_rate"] == 1.0
    assert scored["dense"]["refusal_accuracy"] == 1.0


def test_unfilled_blind_rating_is_rejected():
    questions = [{"id": "q1", "question": "问？", "type": "single_concept",
                  "gold_answer": "答。", "required_evidence_groups": [["a"]]}]
    rows = [{"question_id": "q1", "mode": "dense", "answer": "答。",
             "citation_chunk_ids": [], "evidence": []}]
    template, key = make_blind_template(questions, rows)
    with pytest.raises(ValueError):
        score_annotations(template, key)


def test_all_selected_questions_are_distinct_and_have_valid_labels():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "data/eval/test.jsonl").read_text(encoding="utf-8").splitlines()]
    dev = [json.loads(line) for line in (root / "data/eval/dev.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len({row["id"] for row in rows}) == 60
    assert Counter(row["type"] for row in rows) == {
        "single_concept": 14, "multi_hop": 26, "comparison": 10, "unanswerable": 10,
    }
    assert all(set(row) == {
        "id", "question", "type", "gold_answer", "required_evidence_groups",
        "review_status", "gold_evidence_languages",
    } for row in rows)
    assert not {row["question"] for row in rows} & {row["question"] for row in dev}
    assert all(row["review_status"] == "pending_author_review" for row in rows)
    for row in rows:
        groups = row["required_evidence_groups"]
        assert bool(groups) != (row["type"] == "unanswerable")
        assert set(row["gold_evidence_languages"]) <= {"zh", "en"}
        assert bool(row["gold_evidence_languages"]) == bool(groups)
        assert all(group and all(isinstance(chunk_id, str) and len(chunk_id) == 24
                                 for chunk_id in group) for group in groups)
