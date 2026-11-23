"""共用双语向量索引的纯向量、一跳和关系感知两跳检索。"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import yaml

from .embed import DenseIndex
from .graph import GraphIndex, GraphPath, _key
from .schema import Evidence

Mode = Literal["dense", "one_hop", "typed_path"]


@dataclass(frozen=True)
class RetrievalConfig:
    dense_weight: float = 0.35
    path_weight: float = 0.50
    coverage_weight: float = 0.15
    hop_penalty: float = 0.80
    type_mismatch_weight: float = 0.65
    one_hop_bonus: float = 0.05
    seed_k: int = 8
    max_per_section: int = 2
    bilingual_duplicate_cosine: float = 0.98
    use_relation_types: bool = True

    def __post_init__(self) -> None:
        weights = (self.dense_weight, self.path_weight, self.coverage_weight,
                   self.hop_penalty, self.type_mismatch_weight,
                   self.one_hop_bonus, self.bilingual_duplicate_cosine)
        if any(not 0 <= weight <= 1 for weight in weights):
            raise ValueError("检索权重必须在 0 到 1 之间")
        if self.seed_k < 1 or self.max_per_section < 1:
            raise ValueError("种子数和同章节上限必须为正整数")


def load_retrieval_config(path: Path) -> RetrievalConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("检索配置必须是映射")
    return RetrievalConfig(**data)


def _estimated_tokens(text: str) -> int:
    """无指定生成 tokenizer 时使用稳定的中英混合近似值。"""
    return len(re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_]+|[^\s]", text))


def _relation_intent(question: str) -> set[str]:
    lower = question.casefold()
    intent = set()
    if any(word in lower for word in ("区别", "对比", "比较", "不同", "contrast", "difference", "versus")):
        intent.add("contrasts_with")
    if any(word in lower for word in ("组成", "包含", "结构", "组件", "部分", "component", "consist")):
        intent.add("component_of")
    if any(word in lower for word in ("前置", "基础", "先学", "依赖", "prerequisite", "require")):
        intent.add("prerequisite_of")
    if any(word in lower for word in ("应用", "使用", "用于", "作用", "used", "apply")):
        intent.add("used_in")
    return intent


class Retriever:
    def __init__(
        self, dense: DenseIndex, graph: GraphIndex | None = None,
        config: RetrievalConfig | None = None,
        *, token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self.dense = dense
        self.graph = graph
        self.config = config or RetrievalConfig()
        self.token_counter = token_counter or _estimated_tokens

    def _path_score(self, path: GraphPath, question: str) -> float:
        assert self.graph is not None
        intent = _relation_intent(question)
        by_identity = {
            (self.graph._concept(edge.subject), edge.predicate,
             self.graph._concept(edge.object), edge.evidence_chunk_id): edge.confidence
            for edge in self.graph.edges
        }
        confidences = []
        for left, right, relation, evidence_id, direction in zip(
            path.nodes, path.nodes[1:], path.relations, path.evidence_chunk_ids,
            path.directions,
        ):
            subject, object_ = (left, right) if direction == "forward" else (right, left)
            confidences.append(by_identity[(subject, relation, object_, evidence_id)])
        confidence = sum(confidences) / len(confidences)
        type_weight = 1.0
        if self.config.use_relation_types and intent:
            type_weight = max(1.0 if relation in intent else self.config.type_mismatch_weight
                              for relation in path.relations)
        return confidence * type_weight * self.config.hop_penalty ** (len(path.relations) - 1)

    def _mentioned_concepts(self, question: str) -> set[str]:
        assert self.graph is not None
        compact = lambda value: "".join(_key(value).split())
        query = compact(question)
        known = {self.graph._concept(edge.subject) for edge in self.graph.edges}
        known.update(self.graph._concept(edge.object) for edge in self.graph.edges)
        mentioned = {concept for concept in known if compact(concept) in query}
        for alias in self.graph.aliases:
            if compact(alias["alias"]) in query:
                mentioned.add(self.graph._concept(alias["alias"]))
        return {
            concept for concept in mentioned
            if not any(compact(concept) != compact(other) and
                       compact(concept) in compact(other)
                       for other in mentioned)
        }

    def _coverage(self, path: GraphPath, question: str) -> float:
        mentioned = self._mentioned_concepts(question)
        if not mentioned:
            return 0.0
        return len(mentioned.intersection(path.nodes)) / len(mentioned)

    def retrieve(self, question: str, mode: Mode, k: int, token_budget: int) -> list[Evidence]:
        if mode not in ("dense", "one_hop", "typed_path"):
            raise ValueError(f"未知检索模式：{mode}")
        if k < 1 or token_budget < 1:
            return []
        if mode != "dense" and self.graph is None:
            raise ValueError("图检索需要图索引")

        # One query encoding supplies every mode with exactly the same cosine scores.
        dense_ranked = self.dense.search(question, len(self.dense.chunks))
        by_id = {item.chunk.chunk_id: item for item in dense_ranked}
        candidates = list(dense_ranked if mode == "dense" else dense_ranked[:self.config.seed_k])
        best_paths: dict[str, tuple[float, GraphPath]] = {}
        if mode != "dense":
            assert self.graph is not None
            seed_ids = {item.chunk.chunk_id for item in candidates}
            seed_concepts = {
                self.graph._concept(label)
                for edge in self.graph.edges if edge.evidence_chunk_id in seed_ids
                for label in (edge.subject, edge.object)
            }
            mentioned_seeds = seed_concepts.intersection(self._mentioned_concepts(question))
            if mentioned_seeds:
                seed_concepts = mentioned_seeds
            for concept in sorted(seed_concepts):
                for path in self.graph.neighbors(concept, 1 if mode == "one_hop" else 2):
                    evidence_id = path.evidence_chunk_ids[-1]
                    if evidence_id not in by_id or evidence_id not in self.graph.chunks:
                        continue
                    score = (self.config.one_hop_bonus if mode == "one_hop"
                             else self._path_score(path, question))
                    previous = best_paths.get(evidence_id)
                    rank = (score if mode == "one_hop" else
                            self.config.path_weight * score +
                            self.config.coverage_weight * self._coverage(path, question))
                    previous_rank = (
                        -1.0 if previous is None else
                        (previous[0] if mode == "one_hop" else
                         self.config.path_weight * previous[0] +
                         self.config.coverage_weight * self._coverage(previous[1], question))
                    )
                    if rank > previous_rank:
                        best_paths[evidence_id] = (score, path)
            candidates = [by_id[id] for id in set(seed_ids) | set(best_paths)]

        scored: list[Evidence] = []
        for item in candidates:
            id = item.chunk.chunk_id
            path = best_paths.get(id)
            normalized_dense = (item.score + 1.0) / 2.0
            if mode == "dense":
                score = normalized_dense
            elif mode == "one_hop":
                score = normalized_dense + (path[0] if path else 0.0)
            else:
                score = self.config.dense_weight * normalized_dense
                if path:
                    score += (self.config.path_weight * path[0] +
                              self.config.coverage_weight * self._coverage(path[1], question))
            scored.append(Evidence(
                item.chunk, score, path[1].nodes if path else (),
                path[1].relations if path else (),
                path[1].evidence_chunk_ids if path else (),
            ))
        scored.sort(key=lambda item: (-item.score, item.chunk.chunk_id))

        selected: list[Evidence] = []
        section_counts: Counter[str] = Counter()
        index_by_id = {chunk.chunk_id: i for i, chunk in enumerate(self.dense.chunks)}
        used_tokens = 0
        for item in scored:
            if len(selected) >= k:
                break
            chunk = item.chunk
            cost = self.token_counter(chunk.text)
            if cost < 0:
                raise ValueError("token 计数不能为负")
            if used_tokens + cost > token_budget:
                continue
            section = _key(chunk.section)
            if section_counts[section] >= self.config.max_per_section:
                continue
            index = index_by_id[chunk.chunk_id]
            if any(
                other.chunk.language != chunk.language and
                _key(other.chunk.section) == section and
                float(np.dot(self.dense.vectors[index],
                             self.dense.vectors[index_by_id[other.chunk.chunk_id]])) >=
                self.config.bilingual_duplicate_cosine
                for other in selected
            ):
                continue
            selected.append(item)
            section_counts[section] += 1
            used_tokens += cost
        return selected
