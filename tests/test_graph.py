"""带原文来源的概念图：无网络、无真实 API。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from kgrag.schema import Chunk


def chunk(chunk_id: str, text: str, language: str, section: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id, source_id=f"source-{language}",
        document_id=f"document-{language}", section=section, text=text,
        language=language, source_path=f"{language}.md", line_start=1,
        line_end=2, url=f"https://example.org/{language}#L1-L2",
    )


class FakeClient:
    model = "fixture/extractor"

    def __init__(self, response: dict) -> None:
        self.response = response
        self.calls: list[list[dict]] = []

    def complete_json(self, messages: list[dict]) -> dict:
        self.calls.append(messages)
        return self.response


def test_keeps_only_valid_edges_and_counts_rejection_reasons(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    chunks = [chunk("zh-1", "注意力机制使用查询和键。", "zh", "注意力")]
    client = FakeClient({"edges": [
        {"subject": "查询", "predicate": "used_in", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.9},
        {"subject": "键", "predicate": "component_of", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.8},
        {"subject": "键", "predicate": "causes", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.9},
        {"subject": "查询", "predicate": "used_in", "object": "模型",
         "evidence_chunk_id": "invented", "confidence": 0.9},
    ]})

    graph = build_graph(chunks, client, tmp_path)

    assert len(graph.edges) == 2
    assert graph.rejected == {"invalid_predicate": 1, "invalid_evidence": 1}
    assert {edge.evidence_chunk_id for edge in graph.edges} == {"zh-1"}
    assert graph.evidence("zh-1") == chunks[0]
    assert graph.neighbors("查询", 1)[0].evidence_chunk_ids == ("zh-1",)


def test_only_reviewed_bilingual_alias_connects_two_hops(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    chunks = [
        chunk("zh-1", "查询是注意力机制的组成部分。", "zh", "注意力"),
        chunk("en-1", "Attention is used in Transformer.", "en", "Attention"),
    ]

    class SectionClient(FakeClient):
        def complete_json(self, messages: list[dict]) -> dict:
            self.calls.append(messages)
            payload = json.loads(messages[-1]["content"])
            if payload["chunks"][0]["language"] == "zh":
                return {"edges": [{"subject": "查询", "predicate": "component_of",
                                   "object": "注意力机制", "evidence_chunk_id": "zh-1",
                                   "confidence": 0.9}]}
            return {"edges": [{"subject": "Attention", "predicate": "used_in",
                               "object": "Transformer", "evidence_chunk_id": "en-1",
                               "confidence": 0.8}]}

    client = SectionClient({})
    graph = build_graph(chunks, client, tmp_path)
    assert all(path.nodes[-1] != "Transformer" for path in graph.neighbors("查询", 2))

    aliases = [{"canonical": "注意力机制", "alias": "Attention",
                "canonical_chunk_id": "zh-1", "alias_chunk_id": "en-1",
                "verified_by": "human-review-1"}]
    graph = build_graph(chunks, client, tmp_path, aliases=aliases)
    paths = graph.neighbors("查询", 2)
    assert [(path.nodes, path.relations, path.evidence_chunk_ids) for path in paths] == [
        (("查询", "注意力机制"), ("component_of",), ("zh-1",)),
        (("查询", "注意力机制", "Transformer"),
         ("component_of", "used_in"), ("zh-1", "en-1")),
    ]
    assert paths[1].directions == ("forward", "forward")
    assert graph.neighbors("Transformer", 1)[0].directions == ("reverse",)
    assert len(client.calls) == 2  # second build reads its extraction cache
    assert len({edge.evidence_chunk_id for edge in graph.edges}) == 2

    reloaded = graph.load(tmp_path)
    assert reloaded.neighbors("查询", 2) == graph.neighbors("查询", 2)
    assert reloaded.evidence("en-1").language == "en"


def test_same_language_case_variants_share_a_concept_without_alias() -> None:
    from kgrag.graph import GraphIndex
    from kgrag.schema import Edge

    chunks = [
        chunk("zh-1", "循环神经网络具有隐状态。", "zh", "循环神经网络"),
        chunk("en-1", "Recurrent neural networks have recurrent layers.", "en", "RNN"),
    ]
    graph = GraphIndex(chunks, [
        Edge("隐状态", "used_in", "Recurrent neural networks", "zh-1", 0.8),
        Edge("recurrent neural networks", "component_of", "recurrent layers", "en-1", 0.9),
    ], [], {})

    paths = graph.neighbors("隐状态", 2)
    assert len(paths) == 2
    assert paths[1].nodes == ("隐状态", "Recurrent neural networks", "recurrent layers")
    assert paths[1].evidence_chunk_ids == ("zh-1", "en-1")


def test_rejects_unverified_or_untraceable_aliases(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    chunks = [
        chunk("zh-1", "注意力机制", "zh", "注意力"),
        chunk("en-1", "Attention", "en", "Attention"),
    ]
    client = FakeClient({"edges": []})
    candidate = {"canonical": "注意力机制", "alias": "Attention",
                 "canonical_chunk_id": "zh-1", "alias_chunk_id": "en-1"}
    with pytest.raises(ValueError, match="审核"):
        build_graph(chunks, client, tmp_path, aliases=[candidate])
    with pytest.raises(ValueError, match="片段"):
        build_graph(chunks, client, tmp_path, aliases=[{**candidate,
                    "alias_chunk_id": "missing", "verified_by": "reviewer"}])


def test_load_refuses_alias_without_review_evidence(tmp_path: Path) -> None:
    from kgrag.graph import GraphIndex, build_graph

    chunks = [
        chunk("zh-1", "注意力机制", "zh", "注意力"),
        chunk("en-1", "Attention", "en", "Attention"),
    ]
    build_graph(chunks, FakeClient({"edges": []}), tmp_path)
    path = tmp_path / "graph.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["aliases"] = [{"canonical": "注意力机制", "alias": "Attention",
                        "canonical_chunk_id": "zh-1", "alias_chunk_id": "missing",
                        "verified_by": "reviewer"}]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="片段"):
        GraphIndex.load(tmp_path)


def test_reviewed_rejection_is_excluded_from_paths_and_persists(tmp_path: Path) -> None:
    from kgrag.graph import GraphIndex, build_graph

    sample = chunk("zh-1", "查询和值用于注意力机制。", "zh", "注意力")
    client = FakeClient({"edges": [
        {"subject": "查询", "predicate": "used_in", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.9},
        {"subject": "值", "predicate": "prerequisite_of", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.9},
    ]})
    reviews = [{"subject": "值", "predicate": "prerequisite_of",
                "object": "注意力机制", "evidence_chunk_id": "zh-1",
                "status": "rejected", "reason": "原文只说明使用，并非前置条件"}]

    graph = build_graph([sample], client, tmp_path, reviews=reviews)

    assert [(edge.subject, edge.predicate) for edge in graph.edges] == [("查询", "used_in")]
    assert graph.neighbors("值", 2) == []
    assert len(client.calls) == 1
    manifest = json.loads((tmp_path / "graph_manifest.json").read_text(encoding="utf-8"))
    assert manifest["review_summary"] == {"rejected": 1, "uncertain": 0, "supported": 0,
                                           "stale": 0}
    reloaded = GraphIndex.load(tmp_path)
    assert reloaded.edges == graph.edges
    assert reloaded.reviews == reviews


def test_batches_by_language_and_section_and_caches_same_input(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    chunks = [
        chunk("zh-1", "查询和键。", "zh", "A"),
        chunk("zh-2", "注意力权重。", "zh", "A"),
        chunk("en-1", "Attention weights.", "en", "B"),
    ]
    client = FakeClient({"edges": []})
    build_graph(chunks, client, tmp_path)
    assert len(client.calls) == 2
    request_groups = [json.loads(messages[-1]["content"])["chunks"] for messages in client.calls]
    assert sorted(len(group) for group in request_groups) == [1, 2]
    assert all(len({item["language"] for item in group}) == 1 for group in request_groups)
    build_graph(list(reversed(chunks)), client, tmp_path)
    assert len(client.calls) == 2
    changed = [chunks[0], chunk("zh-2", "修订后的权重。", "zh", "A"), chunks[2]]
    build_graph(changed, client, tmp_path)
    assert len(client.calls) == 3


def test_extraction_settings_are_recorded_and_change_cache_key(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    sample = chunk("zh-1", "查询用于注意力机制。", "zh", "注意力")
    client = FakeClient({"edges": []})
    client.temperature = 0.0
    client.max_tokens = 600
    client.base_url = "https://api.example.org"
    build_graph([sample], client, tmp_path)
    client.temperature = 0.5
    build_graph([sample], client, tmp_path)
    manifest = json.loads((tmp_path / "graph_manifest.json").read_text(encoding="utf-8"))
    assert len(client.calls) == 2
    assert manifest["extraction_temperature"] == 0.5
    assert manifest["extraction_max_tokens"] == 600


def test_rejects_out_of_batch_evidence_and_nonfinite_confidence(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    chunks = [
        chunk("zh-1", "查询。", "zh", "A"),
        chunk("en-1", "Query.", "en", "B"),
    ]

    class WrongClient(FakeClient):
        def complete_json(self, messages: list[dict]) -> dict:
            payload = json.loads(messages[-1]["content"])
            own = payload["chunks"][0]["chunk_id"]
            other = "en-1" if own == "zh-1" else "zh-1"
            return {"edges": [
                {"subject": "A", "predicate": "used_in", "object": "B",
                 "evidence_chunk_id": other, "confidence": 0.8},
                {"subject": "A", "predicate": "used_in", "object": "B",
                 "evidence_chunk_id": own, "confidence": float("nan")},
            ]}

    graph = build_graph(chunks, WrongClient({}), tmp_path)
    assert graph.edges == []
    assert graph.rejected == {"invalid_evidence": 2, "invalid_confidence": 2}


def test_malformed_nested_fields_do_not_abort_other_edges(tmp_path: Path) -> None:
    from kgrag.graph import build_graph

    sample = chunk("zh-1", "查询用于注意力机制。", "zh", "注意力")
    client = FakeClient({"edges": [
        {"subject": "坏", "predicate": ["used_in"], "object": "关系",
         "evidence_chunk_id": "zh-1", "confidence": 0.7},
        {"subject": "坏", "predicate": "used_in", "object": "关系",
         "evidence_chunk_id": {"id": "zh-1"}, "confidence": 0.7},
        {"subject": "查询", "predicate": "used_in", "object": "注意力机制",
         "evidence_chunk_id": "zh-1", "confidence": 0.9},
    ]})
    graph = build_graph([sample], client, tmp_path)
    assert len(graph.edges) == 1
    assert graph.rejected == {"invalid_predicate": 1, "invalid_evidence": 1}


def test_cli_builds_graph_from_chunks_and_records_source_revision(tmp_path: Path, capsys) -> None:
    from kgrag.cli import main

    chunks = tmp_path / "chunks.jsonl"
    sample = chunk("zh-1", "查询用于注意力机制。", "zh", "注意力")
    chunks.write_text(json.dumps(asdict(sample), ensure_ascii=False) + "\n", encoding="utf-8")
    revision = "a" * 40
    manifest = {"sources": [{"source_id": sample.source_id, "revision": revision}]}
    (tmp_path / "build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("aliases: []\n", encoding="utf-8")
    out_dir = tmp_path / "graph"
    client = FakeClient({"edges": [{
        "subject": "查询", "predicate": "used_in", "object": "注意力机制",
        "evidence_chunk_id": "zh-1", "confidence": 0.9,
    }]})
    with patch("kgrag.cli.DeepSeekChatClient", return_value=client):
        assert main(["build-graph", "--chunks", str(chunks), "--out-dir", str(out_dir),
                     "--aliases", str(aliases), "--chat-model", client.model]) == 0
    assert "有效关系 1" in capsys.readouterr().out
    result = json.loads((out_dir / "graph_manifest.json").read_text(encoding="utf-8"))
    assert result["corpus_sources"] == manifest["sources"]
    assert result["prompt_version"]


def test_cli_applies_curated_review_file(tmp_path: Path, capsys) -> None:
    from kgrag.cli import main
    from kgrag.graph import GraphIndex, PROMPT_VERSION

    sample = chunk("zh-1", "查询用于注意力机制。", "zh", "注意力")
    chunks = tmp_path / "chunks.jsonl"
    chunks.write_text(json.dumps(asdict(sample), ensure_ascii=False) + "\n", encoding="utf-8")
    aliases = tmp_path / "aliases.yaml"
    aliases.write_text("aliases: []\n", encoding="utf-8")
    reviews = tmp_path / "reviews.yaml"
    reviews.write_text(
        f"prompt_version: {PROMPT_VERSION}\nreviews:\n"
        "  - subject: 查询\n    predicate: used_in\n    object: 注意力机制\n"
        "    evidence_chunk_id: zh-1\n    status: rejected\n    reason: 审核拒绝\n",
        encoding="utf-8",
    )
    client = FakeClient({"edges": [{
        "subject": "查询", "predicate": "used_in", "object": "注意力机制",
        "evidence_chunk_id": "zh-1", "confidence": 0.9,
    }]})
    out_dir = tmp_path / "graph"
    with patch("kgrag.cli.DeepSeekChatClient", return_value=client):
        assert main(["build-graph", "--chunks", str(chunks), "--out-dir", str(out_dir),
                     "--aliases", str(aliases), "--reviews", str(reviews)]) == 0
    graph = GraphIndex.load(out_dir)
    assert graph.edges == []
    assert graph.neighbors("查询", 2) == []
    assert "审核排除 1" in capsys.readouterr().out
