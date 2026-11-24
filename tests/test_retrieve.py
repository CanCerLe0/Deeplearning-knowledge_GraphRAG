"""三路检索共用向量空间、证据来源和上下文预算。"""

from __future__ import annotations

import numpy as np
import json
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from kgrag.embed import DenseIndex
from kgrag.graph import GraphIndex
from kgrag.schema import Chunk, Edge


class QueryEmbedder:
    model_id = "fixture"
    revision = "v1"

    def encode(self, texts):
        assert len(texts) == 1
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def chunk(id: str, text: str, language: str, section: str) -> Chunk:
    return Chunk(id, "source-" + language, "doc-" + id, section, text,
                 language, language + ".md", 1, 2, None)


def fixture():
    chunks = [
        chunk("seed", "隐状态不同于隐藏层。", "zh", "循环网络"),
        chunk("near", "隐状态是循环网络的内部信息。", "zh", "循环网络"),
        chunk("bridge", "隐状态用于循环神经网络。", "zh", "循环网络结构"),
        chunk("target", "Recurrent layers maintain hidden state.", "en", "RNN layers"),
        chunk("unrelated", "Unrelated dense match.", "en", "Other"),
    ]
    vectors = np.asarray([
        [1.0, 0.0], [0.98, 0.2], [0.6, 0.8], [0.0, 1.0], [0.97, 0.24],
    ], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    dense = DenseIndex(chunks, vectors, QueryEmbedder())
    aliases = [{"canonical": "隐状态", "alias": "hidden state"}]
    graph = GraphIndex(chunks, [
        Edge("隐藏层", "contrasts_with", "隐状态", "seed", 0.9),
        Edge("隐状态", "component_of", "Recurrent neural networks", "bridge", 0.9),
        Edge("recurrent neural networks", "component_of", "recurrent layers", "target", 0.95),
    ], aliases, {})
    return dense, graph


def ids(results):
    return [item.chunk.chunk_id for item in results]


def test_typed_path_can_surface_two_hop_english_evidence_from_chinese_seed():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, graph = fixture()
    retriever = Retriever(dense, graph, RetrievalConfig(seed_k=2),
                          token_counter=lambda text: len(text))
    query = "隐状态与循环层有什么关系？"
    dense_results = retriever.retrieve(query, "dense", 3, 2000)
    one_hop = retriever.retrieve(query, "one_hop", 5, 2000)
    typed = retriever.retrieve(query, "typed_path", 3, 2000)

    assert "target" not in ids(dense_results)
    assert "target" not in ids(one_hop)
    assert all(len(item.path_relations) <= 1 for item in one_hop)
    assert "target" in ids(typed)
    target = next(item for item in typed if item.chunk.chunk_id == "target")
    assert target.chunk.language == "en"
    assert target.path == ("隐藏层", "隐状态", "Recurrent neural networks", "recurrent layers") or target.path == ("隐状态", "Recurrent neural networks", "recurrent layers")
    assert target.path_relations[-1] == "component_of"
    assert target.path_evidence_chunk_ids[-1] == "target"


def test_seed_text_can_start_graph_expansion_without_carrying_edge_evidence():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [
        chunk("seed", "隐状态是循环网络的内部信息。", "zh", "循环网络"),
        chunk("target", "Hidden state is used in recurrent layers.", "en", "RNN layers"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0., 1.]]), QueryEmbedder())
    graph = GraphIndex(chunks, [
        Edge("隐状态", "used_in", "recurrent layers", "target", 0.95),
    ], [], {})

    results = Retriever(dense, graph, RetrievalConfig(seed_k=1)).retrieve(
        "隐状态用于什么？", "typed_path", 2, 2000)
    assert "target" in ids(results)
    target = next(item for item in results if item.chunk.chunk_id == "target")
    assert target.path == ("隐状态", "recurrent layers")


def test_dense_mode_does_not_access_graph():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, _ = fixture()
    assert ids(Retriever(dense).retrieve("隐状态与循环层有什么关系？", "dense", 2, 2000)) == ["seed", "near"]


def test_budget_dedup_and_cross_language_near_duplicate_limit():
    from kgrag.answer import build_answer_messages
    from kgrag.retrieve import Retriever

    dense, graph = fixture()
    duplicate = chunk("translation", "Hidden state differs from hidden layer.", "en", "循环网络")
    dense.chunks.append(duplicate)
    dense.vectors = np.vstack([dense.vectors, dense.vectors[0]])
    retriever = Retriever(dense, graph, token_counter=lambda text: len(text))
    query = "隐状态与循环层有什么关系？"
    results = retriever.retrieve(query, "dense", 6, 2000)

    assert results
    assert len(ids(results)) == len(set(ids(results)))
    assert sum(len(message["content"]) for message in build_answer_messages(query, results)) <= 2000
    assert not {"seed", "translation"}.issubset(ids(results))
    for mode in ("one_hop", "typed_path"):
        selected = retriever.retrieve(query, mode, 6, 2000)
        assert selected
        assert sum(len(message["content"]) for message in build_answer_messages(query, selected)) <= 2000


def test_every_mode_limits_the_actual_answer_messages_including_metadata():
    from kgrag.answer import generate_answer
    from kgrag.retrieve import RetrievalConfig, Retriever

    url = "https://example.org/" + "x" * 80
    chunks = [
        Chunk("a", "source", "doc", "Intro", "Alpha", "en", "a.md", 1, 1, url),
        Chunk("b", "source", "doc", "Intro", "Beta", "en", "b.md", 1, 1, url),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0.8, 0.2]]), QueryEmbedder())
    graph = GraphIndex(chunks, [Edge("Alpha", "used_in", "Beta", "b", 0.95)], [], {})

    class RecordingClient:
        def complete_json(self, messages):
            self.messages = messages
            cited = json.loads(messages[1]["content"])["evidence"][0]["chunk_id"]
            return {"answer": f"Fact.[{cited}]", "citation_chunk_ids": [cited]}

    for mode in ("dense", "one_hop", "typed_path"):
        selected = Retriever(dense, graph, RetrievalConfig(seed_k=1),
                             token_counter=len).retrieve("Question?", mode, 2, 450)
        assert len(selected) == 1
        client = RecordingClient()
        generate_answer("Question?", selected, client)
        assert sum(len(message["content"]) for message in client.messages) <= 450


def test_graph_edge_without_indexed_evidence_never_enters_results():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, graph = fixture()
    graph.edges.append(Edge("隐状态", "used_in", "fictional", "missing", 1.0))
    results = Retriever(dense, graph, RetrievalConfig(seed_k=2)).retrieve(
        "隐状态与循环层有什么关系？", "typed_path", 5, 2000)
    assert "missing" not in ids(results)


def test_two_hop_path_requires_indexed_evidence_for_every_edge():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [
        chunk("seed", "概念乙的基础说明。", "zh", "基础"),
        chunk("target", "Concept D is explained here.", "en", "Advanced"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0., 1.]]), QueryEmbedder())
    graph = GraphIndex(chunks, [
        Edge("甲", "used_in", "乙", "seed", 0.9),
        Edge("乙", "used_in", "丙", "missing", 1.0),
        Edge("丙", "used_in", "丁", "target", 1.0),
    ], [], {})

    results = Retriever(dense, graph, RetrievalConfig(seed_k=1)).retrieve(
        "乙如何用于丁？", "typed_path", 2, 2000)
    assert ids(results) == ["seed"]


def test_bilingual_near_duplicates_with_different_section_titles_share_limit():
    from kgrag.retrieve import Retriever

    chunks = [
        chunk("zh", "隐状态用于循环层。", "zh", "循环网络"),
        chunk("en", "Hidden state in recurrent layers.", "en", "Recurrent Neural Networks"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [1., 0.]]), QueryEmbedder())

    results = Retriever(dense).retrieve("隐状态与循环层", "dense", 2, 2000)
    assert len(results) == 1


def test_section_limit_does_not_combine_distinct_documents():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [chunk(str(i), f"Distinct evidence {i}.", "en", "Introduction")
              for i in range(3)]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0.9, 0.1], [0.8, 0.2]]),
                       QueryEmbedder())

    results = Retriever(dense, config=RetrievalConfig(max_per_section=2)).retrieve(
        "question", "dense", 3, 1000)
    assert ids(results) == ["0", "1", "2"]


def test_section_limit_still_applies_inside_one_document():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [chunk(str(i), f"Distinct evidence {i}.", "en", "Introduction")
              for i in range(3)]
    chunks = [Chunk(c.chunk_id, c.source_id, "shared-document", c.section, c.text,
                    c.language, c.source_path, c.line_start, c.line_end, c.url)
              for c in chunks]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0.9, 0.1], [0.8, 0.2]]),
                       QueryEmbedder())

    results = Retriever(dense, config=RetrievalConfig(max_per_section=2)).retrieve(
        "question", "dense", 3, 1000)
    assert ids(results) == ["0", "1"]


def test_path_ablation_removes_graph_boost():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, graph = fixture()
    results = Retriever(dense, graph, RetrievalConfig(path_weight=0, seed_k=2)).retrieve(
        "隐状态与循环层有什么关系？", "typed_path", 3, 2000)
    assert "target" not in ids(results)


def test_unmentioned_seed_concept_cannot_boost_unrelated_evidence():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, graph = fixture()
    graph.edges.append(Edge("decoder attention", "component_of", "masked attention", "near", 0.99))
    results = Retriever(dense, graph, RetrievalConfig(seed_k=2)).retrieve(
        "隐状态与循环层有什么关系？", "typed_path", 5, 2000)
    near = next(item for item in results if item.chunk.chunk_id == "near")
    assert near.path == ()


def test_spaced_question_concept_prefers_specific_encoder_path():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [
        chunk("seed", "层规范化用于模型。", "zh", "seed"),
        chunk("generic", "Layer norm in Transformer.", "en", "generic"),
        chunk("specific", "层规范化用于 Transformer 编码器。", "zh", "specific"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0.4, 0.9], [0.4, 0.9]]), QueryEmbedder())
    graph = GraphIndex(chunks, [
        Edge("层规范化", "component_of", "Transformer", "generic", 0.9),
        Edge("层规范化", "component_of", "Transformer编码器", "specific", 0.9),
        Edge("层规范化", "used_in", "模型", "seed", 0.8),
    ], [], {})
    results = Retriever(dense, graph, RetrievalConfig(seed_k=1)).retrieve(
        "层规范化在 Transformer 编码器中起什么作用？", "typed_path", 3, 2000)
    assert ids(results).index("specific") < ids(results).index("generic")


def test_best_path_uses_query_coverage_as_well_as_edge_confidence():
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [
        chunk("seed", "层规范化与 Transformer 编码器。", "zh", "seed"),
        chunk("target", "层规范化和前馈网络都在 Transformer 编码器中。", "zh", "target"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0.4, 0.9]]), QueryEmbedder())
    graph = GraphIndex(chunks, [
        Edge("层规范化", "used_in", "Transformer编码器", "seed", 0.8),
        Edge("层规范化", "component_of", "Transformer编码器", "target", 0.9),
        Edge("Transformer编码器", "component_of", "前馈网络", "target", 0.95),
    ], [], {})
    results = Retriever(dense, graph, RetrievalConfig(seed_k=1)).retrieve(
        "层规范化与 Transformer 编码器有什么关系？", "typed_path", 2, 2000)
    target = next(item for item in results if item.chunk.chunk_id == "target")
    assert len(target.path) == 2
    assert set(target.path) == {"层规范化", "Transformer编码器"}


def test_relation_type_ablation_changes_mismatched_path_score():
    from kgrag.retrieve import RetrievalConfig, Retriever

    dense, graph = fixture()
    query = "隐状态和隐藏层有什么区别？"
    weighted = Retriever(dense, graph, RetrievalConfig(seed_k=2)).retrieve(query, "typed_path", 5, 2000)
    unweighted = Retriever(dense, graph, RetrievalConfig(seed_k=2, use_relation_types=False)).retrieve(
        query, "typed_path", 5, 2000)
    weighted_target = next(item.score for item in weighted if item.chunk.chunk_id == "target")
    unweighted_target = next(item.score for item in unweighted if item.chunk.chunk_id == "target")
    assert weighted_target < unweighted_target


def test_cli_ask_uses_typed_path_and_reports_provenance(tmp_path):
    from kgrag.cli import main

    dense, graph = fixture()

    class FakeClient:
        last_model = "fixture-v1"
        last_usage = {}

        def __init__(self, **kwargs):
            pass

        def complete_json(self, messages):
            return {"answer": "循环层维持隐状态。[target]", "citation_chunk_ids": ["target"]}

    output = StringIO()
    with patch("kgrag.cli.FlagEmbedder", return_value=QueryEmbedder()), patch(
        "kgrag.cli.DenseIndex.load", return_value=dense
    ), patch("kgrag.cli.GraphIndex.load", return_value=graph), patch(
        "kgrag.cli.DeepSeekChatClient", FakeClient
    ), redirect_stdout(output):
        assert main(["ask", "隐状态与循环层有什么关系？", "--mode", "typed_path",
                     "--top-k", "3", "--token-budget", "2000",
                     "--index-dir", str(tmp_path), "--graph-dir", str(tmp_path)]) == 0
    result = json.loads(output.getvalue())
    target = next(item for item in result["evidence"] if item["chunk_id"] == "target")
    assert target["language"] == "en"
    assert target["path"][-1] == "recurrent layers"
    assert target["path_evidence_chunk_ids"][-1] == "target"
    assert result["run_config"]["retrieval_mode"] == "typed_path"


def test_reverse_traversal_keeps_original_relation_direction_in_evidence_and_cli(tmp_path):
    from kgrag.cli import main
    from kgrag.retrieve import RetrievalConfig, Retriever

    chunks = [
        chunk("seed", "循环神经网络的结构。", "zh", "循环网络"),
        chunk("target", "Hidden state is a component of recurrent neural networks.", "en", "RNN"),
    ]
    dense = DenseIndex(chunks, np.asarray([[1., 0.], [0., 1.]]), QueryEmbedder())
    graph = GraphIndex(chunks, [
        Edge("隐状态", "component_of", "循环神经网络", "target", 0.95),
    ], [], {})
    query = "循环神经网络中的隐状态是什么？"
    results = Retriever(dense, graph, RetrievalConfig(seed_k=1)).retrieve(
        query, "one_hop", 2, 2000)
    target = next(item for item in results if item.chunk.chunk_id == "target")
    assert target.path == ("循环神经网络", "隐状态")
    assert target.path_relations == ("component_of",)
    assert target.path_directions == ("reverse",)

    class FakeClient:
        last_model = "fixture-v1"
        last_usage = {}

        def __init__(self, **kwargs):
            pass

        def complete_json(self, messages):
            return {"answer": "隐状态是网络的一部分。[target]", "citation_chunk_ids": ["target"]}

    output = StringIO()
    with patch("kgrag.cli.FlagEmbedder", return_value=QueryEmbedder()), patch(
        "kgrag.cli.DenseIndex.load", return_value=dense
    ), patch("kgrag.cli.GraphIndex.load", return_value=graph), patch(
        "kgrag.cli.DeepSeekChatClient", FakeClient
    ), redirect_stdout(output):
        assert main(["ask", query, "--mode", "one_hop", "--top-k", "2",
                     "--token-budget", "2000", "--index-dir", str(tmp_path),
                     "--graph-dir", str(tmp_path)]) == 0
    payload = json.loads(output.getvalue())
    target_json = next(item for item in payload["evidence"] if item["chunk_id"] == "target")
    assert target_json["path_directions"] == ["reverse"]
