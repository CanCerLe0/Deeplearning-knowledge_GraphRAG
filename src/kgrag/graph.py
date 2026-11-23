"""从双语原文片段抽取、校验并保存可追溯的概念图。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .llm import ChatClient
from .schema import Chunk, Edge


PROMPT_VERSION = "concept-relations-v1"
PREDICATES = frozenset({"prerequisite_of", "component_of", "contrasts_with", "used_in"})


def _name(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _key(value: str) -> str:
    return _name(value).casefold()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}-",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class GraphPath:
    """概念路径与逐跳原文证据；节点名使用已审核的标准名。"""

    nodes: tuple[str, ...]
    relations: tuple[str, ...]
    evidence_chunk_ids: tuple[str, ...]
    directions: tuple[str, ...] = ()


class GraphIndex:
    def __init__(
        self, chunks: Sequence[Chunk], edges: Sequence[Edge],
        aliases: Sequence[dict[str, str]], rejected: Mapping[str, int],
    ) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.edges = list(edges)
        self.aliases = list(aliases)
        self.rejected = dict(rejected)
        self._canonical: dict[str, str] = {}
        for alias in aliases:
            self._canonical[_key(alias["alias"])] = alias["canonical"]
            self._canonical[_key(alias["canonical"])] = alias["canonical"]

    def evidence(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def _concept(self, value: str) -> str:
        return self._canonical.get(_key(value), _name(value))

    def neighbors(self, concept: str, max_hops: int) -> list[GraphPath]:
        if not 0 <= max_hops <= 2:
            raise ValueError("第一版最多支持两跳")
        start = self._concept(concept)
        adjacency: dict[str, list[tuple[str, Edge, str]]] = defaultdict(list)
        for edge in self.edges:
            subject, object_ = self._concept(edge.subject), self._concept(edge.object)
            if subject == object_:
                continue
            adjacency[subject].append((object_, edge, "forward"))
            adjacency[object_].append((subject, edge, "reverse"))
        paths: list[GraphPath] = []
        queue = deque([GraphPath((start,), (), ())])
        while queue:
            path = queue.popleft()
            if len(path.relations) == max_hops:
                continue
            for next_node, edge, direction in sorted(
                adjacency.get(path.nodes[-1], []),
                key=lambda pair: (pair[0], pair[1].predicate, pair[1].evidence_chunk_id),
            ):
                if next_node in path.nodes:
                    continue
                next_path = GraphPath(
                    path.nodes + (next_node,),
                    path.relations + (edge.predicate,),
                    path.evidence_chunk_ids + (edge.evidence_chunk_id,),
                    path.directions + (direction,),
                )
                paths.append(next_path)
                queue.append(next_path)
        return paths

    def save(self, out_dir: Path, *, manifest: dict[str, Any]) -> None:
        out_dir = Path(out_dir)
        _atomic_json(out_dir / "graph.json", {
            "format_version": 1,
            "chunks": [asdict(chunk) for chunk in self.chunks.values()],
            "edges": [asdict(edge) for edge in self.edges],
            "aliases": self.aliases,
            "rejected": self.rejected,
        })
        _atomic_json(out_dir / "graph_manifest.json", manifest)

    @classmethod
    def load(cls, out_dir: Path) -> GraphIndex:
        data = json.loads((Path(out_dir) / "graph.json").read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("不支持的图索引版本")
        chunks = [Chunk(**item) for item in data["chunks"]]
        edges = [Edge(**item) for item in data["edges"]]
        known = {chunk.chunk_id for chunk in chunks}
        for edge in edges:
            validated, reason = _validated_edge(asdict(edge), known)
            if reason is not None or validated != edge:
                raise ValueError("图索引包含无效关系或无法追溯的证据片段")
        aliases = _validated_aliases(data["aliases"], {chunk.chunk_id: chunk for chunk in chunks})
        return cls(chunks, edges, aliases, data["rejected"])


def _validated_aliases(
    aliases: Sequence[Mapping[str, str]], chunks: Mapping[str, Chunk],
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    known: dict[str, str] = {}
    for record in aliases:
        if not isinstance(record, Mapping):
            raise ValueError("别名审核记录必须是对象")
        canonical = _name(str(record.get("canonical", "")))
        alias = _name(str(record.get("alias", "")))
        reviewer = _name(str(record.get("verified_by", "")))
        if not canonical or not alias or not reviewer:
            raise ValueError("别名需要标准名、别名和审核人")
        if _key(canonical) == _key(alias):
            raise ValueError("别名不能与标准名相同")
        canonical_id = record.get("canonical_chunk_id", "")
        alias_id = record.get("alias_chunk_id", "")
        if canonical_id not in chunks or alias_id not in chunks:
            raise ValueError("别名引用了不存在的片段")
        source, target = chunks[canonical_id], chunks[alias_id]
        if source.language == target.language:
            raise ValueError("跨语言别名必须关联不同语言的片段")
        if _key(canonical) not in _key(source.text) or _key(alias) not in _key(target.text):
            raise ValueError("别名没有出现在对应的原文片段")
        for label in (canonical, alias):
            existing = known.get(_key(label))
            if existing is not None and existing != canonical:
                raise ValueError("同一概念别名映射到多个标准名")
            known[_key(label)] = canonical
        result.append({
            "canonical": canonical, "alias": alias,
            "canonical_chunk_id": canonical_id, "alias_chunk_id": alias_id,
            "verified_by": reviewer,
        })
    return result


def _validated_edge(item: Any, batch_ids: set[str]) -> tuple[Edge | None, str | None]:
    if not isinstance(item, dict):
        return None, "invalid_shape"
    predicate = item.get("predicate")
    if not isinstance(predicate, str) or predicate not in PREDICATES:
        return None, "invalid_predicate"
    evidence_id = item.get("evidence_chunk_id")
    if not isinstance(evidence_id, str) or evidence_id not in batch_ids:
        return None, "invalid_evidence"
    subject, object_ = item.get("subject"), item.get("object")
    if not isinstance(subject, str) or not isinstance(object_, str):
        return None, "invalid_concept"
    subject, object_ = _name(subject), _name(object_)
    if not subject or not object_ or _key(subject) == _key(object_):
        return None, "invalid_concept"
    confidence = item.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None, "invalid_confidence"
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None, "invalid_confidence"
    return Edge(subject, predicate, object_, item["evidence_chunk_id"], float(confidence)), None


def build_graph(
    chunks: list[Chunk], client: ChatClient, out_dir: Path, *,
    aliases: Sequence[Mapping[str, str]] = (), batch_size: int = 8,
    corpus_manifest: Mapping[str, Any] | None = None,
) -> GraphIndex:
    """按语言和小节抽取候选边，校验后保存图与抽取缓存。"""
    if batch_size < 1:
        raise ValueError("batch_size 必须为正整数")
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    if len(by_id) != len(chunks):
        raise ValueError("片段 ID 重复")
    reviewed_aliases = _validated_aliases(aliases, by_id)
    out_dir = Path(out_dir)
    batches: dict[tuple[str, str], list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        batches[(chunk.language, chunk.section)].append(chunk)
    model = str(getattr(client, "model", type(client).__name__))
    extraction_config = {
        "model": model,
        "base_url": getattr(client, "base_url", None),
        "temperature": getattr(client, "temperature", None),
        "max_tokens": getattr(client, "max_tokens", None),
    }
    rejected: Counter[str] = Counter()
    edges: dict[tuple[str, str, str, str], Edge] = {}
    for group in sorted(batches):
        ordered = sorted(batches[group], key=lambda chunk: chunk.chunk_id)
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset:offset + batch_size]
            request = {
                "prompt_version": PROMPT_VERSION,
                "extraction_config": extraction_config,
                "chunks": [asdict(chunk) for chunk in batch],
            }
            cache_key = hashlib.sha256(json.dumps(
                request, ensure_ascii=False, sort_keys=True,
            ).encode("utf-8")).hexdigest()
            cache_path = out_dir / "cache" / f"{cache_key}.json"
            if cache_path.exists():
                response = json.loads(cache_path.read_text(encoding="utf-8"))
            else:
                messages = [
                    {"role": "system", "content": (
                        "从给定 Markdown 片段抽取有原文支持的概念关系。仅返回 JSON 对象，"
                        "格式为 {\"edges\": [{\"subject\": str, \"predicate\": str, "
                        "\"object\": str, \"evidence_chunk_id\": str, \"confidence\": number}]}。"
                        "predicate 仅可为 prerequisite_of、component_of、contrasts_with、used_in。"
                        "每条边必须由同一批次中的一个片段直接支持；不能编造证据 ID。"
                        "若无可靠关系返回空数组。概念保持原文语言，不自行合并译名。"
                    )},
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ]
                response = client.complete_json(messages)
                if isinstance(response, dict) and isinstance(response.get("edges"), list):
                    try:
                        _atomic_json(cache_path, response)
                    except (TypeError, ValueError):
                        # 不规范的模型输出仍交给下方逐条校验，但不写入缓存。
                        pass
            if not isinstance(response, dict) or not isinstance(response.get("edges"), list):
                rejected["invalid_response"] += 1
                continue
            batch_ids = {chunk.chunk_id for chunk in batch}
            for candidate in response["edges"]:
                edge, reason = _validated_edge(candidate, batch_ids)
                if reason is not None:
                    rejected[reason] += 1
                else:
                    assert edge is not None
                    identity = (edge.subject, edge.predicate, edge.object, edge.evidence_chunk_id)
                    previous = edges.get(identity)
                    if previous is None or edge.confidence > previous.confidence:
                        edges[identity] = edge
    graph = GraphIndex(
        sorted(chunks, key=lambda chunk: chunk.chunk_id),
        [edges[key] for key in sorted(edges)], reviewed_aliases, rejected,
    )
    source_revisions = sorted({
        chunk.url.split("/blob/", 1)[1].split("/", 1)[0]
        for chunk in chunks if chunk.url and "/blob/" in chunk.url
    })
    manifest = {
        "format_version": 1, "prompt_version": PROMPT_VERSION,
        "extraction_model_id": model, "chunk_count": len(chunks),
        "extraction_model_version": getattr(client, "last_model", model),
        "extraction_base_url": extraction_config["base_url"],
        "extraction_temperature": extraction_config["temperature"],
        "extraction_max_tokens": extraction_config["max_tokens"],
        "edge_count": len(graph.edges), "alias_count": len(reviewed_aliases),
        "rejected": dict(rejected), "source_revisions": source_revisions,
        "corpus_sources": list(corpus_manifest.get("sources", [])) if corpus_manifest else [],
    }
    graph.save(out_dir, manifest=manifest)
    return graph
