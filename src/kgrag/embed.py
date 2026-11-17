"""统一双语余弦向量索引与可替换的本地 embedding。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .schema import Chunk, Evidence


class Embedder(Protocol):
    model_id: str
    revision: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class FlagEmbedder:
    """通过 FlagEmbedding 运行本地 BGE 模型，默认使用 BGE-M3。"""

    def __init__(
        self, model_id: str = "BAAI/bge-m3", revision: str = "main", *,
        device: str | None = None,
    ) -> None:
        try:
            from FlagEmbedding import FlagAutoModel
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("本地 embedding 需要安装：pip install -e '.[embedding]'") from error
        self.model_id = model_id
        local_path = Path(model_id)
        if local_path.exists():
            model_path = str(local_path)
            self.revision = revision
        else:
            model_path = snapshot_download(repo_id=model_id, revision=revision)
            self.revision = Path(model_path).name
        options = {"use_fp16": device is not None and device.startswith("cuda")}
        if device is not None:
            options["devices"] = device
        self._model = FlagAutoModel.from_finetuned(model_path, **options)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        result = self._model.encode(list(texts))
        if isinstance(result, dict):
            result = result["dense_vecs"]
        return np.asarray(result, dtype=np.float32)


def load_chunks(path: Path) -> list[Chunk]:
    """读取 Task 1 输出的 chunks.jsonl。"""
    return [Chunk(**json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def _unit_vectors(values: np.ndarray, count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != count or values.shape[1] == 0:
        raise ValueError("embedding 维度或记录数不正确")
    if not np.isfinite(values).all():
        raise ValueError("embedding 包含非有限数值")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding 不能是零向量")
    return values / norms


class DenseIndex:
    """同一向量空间中的中英文片段，使用 NumPy 余弦相似度检索。"""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder: Embedder) -> None:
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, out_dir: Path) -> DenseIndex:
        if not chunks:
            raise ValueError("不能从空语料构建向量索引")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("片段 ID 重复")
        vectors = _unit_vectors(embedder.encode([chunk.text for chunk in chunks]), len(chunks))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "dense_vectors.npy", vectors)
        with (out_dir / "dense_chunks.jsonl").open("w", encoding="utf-8") as stream:
            for chunk in chunks:
                stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
        manifest = {
            "embedding_model_id": embedder.model_id,
            "embedding_revision": embedder.revision,
            "normalization": "l2",
            "chunk_count": len(chunks),
            "dimensions": vectors.shape[1],
        }
        (out_dir / "dense_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return cls(chunks, vectors, embedder)

    @classmethod
    def load(cls, out_dir: Path, embedder: Embedder) -> DenseIndex:
        out_dir = Path(out_dir)
        manifest = json.loads((out_dir / "dense_manifest.json").read_text(encoding="utf-8"))
        if (manifest["embedding_model_id"], manifest["embedding_revision"]) != (
            embedder.model_id, embedder.revision
        ):
            raise ValueError("embedding 模型或版本与索引不一致")
        chunks = load_chunks(out_dir / "dense_chunks.jsonl")
        vectors = np.load(out_dir / "dense_vectors.npy", allow_pickle=False)
        if vectors.shape != (manifest["chunk_count"], manifest["dimensions"]):
            raise ValueError("向量索引与构建清单不一致")
        return cls(chunks, vectors, embedder)

    def search(self, query: str, k: int) -> list[Evidence]:
        if k < 1:
            return []
        query_vector = _unit_vectors(self.embedder.encode([query]), 1)[0]
        if query_vector.shape[0] != self.vectors.shape[1]:
            raise ValueError("查询 embedding 维度与索引不一致")
        scores = self.vectors @ query_vector
        indices = np.argsort(-scores, kind="stable")[:k]
        return [Evidence(self.chunks[int(i)], float(scores[i]), ()) for i in indices]
