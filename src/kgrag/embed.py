"""统一双语余弦向量索引与可替换的本地 embedding。"""

from __future__ import annotations

import json
import hashlib
import io
import os
import tempfile
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
        device: str | None = None, model_class: str | None = None,
        pooling_method: str | None = None,
    ) -> None:
        try:
            from FlagEmbedding import FlagAutoModel
            from huggingface_hub import snapshot_download
        except ImportError as error:
            raise RuntimeError("本地 embedding 需要安装：pip install -e '.[embedding]'") from error
        self.model_id = model_id
        self.model_class = model_class or (
            "encoder-only-m3" if Path(model_id).name == "bge-m3" else None
        )
        if self.model_class is None:
            raise ValueError("其他 embedding 模型须显式指定 model_class")
        self.pooling_method = pooling_method
        local_path = Path(model_id)
        if local_path.exists():
            model_path = str(local_path)
            self.revision = revision
        else:
            model_path = snapshot_download(repo_id=model_id, revision=revision)
            self.revision = Path(model_path).name
        options = {
            "model_class": self.model_class,
            "use_fp16": device is not None and device.startswith("cuda"),
        }
        if device is not None:
            options["devices"] = device
        if pooling_method is not None:
            options["pooling_method"] = pooling_method
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


def _write_atomic(path: Path, data: bytes) -> None:
    """先写同目录临时文件，再替换目标文件。"""
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        vector_stream = io.BytesIO()
        np.save(vector_stream, vectors)
        vector_data = vector_stream.getvalue()
        chunk_data = "".join(
            json.dumps(asdict(chunk), ensure_ascii=False) + "\n" for chunk in chunks
        ).encode("utf-8")
        manifest = {
            "embedding_model_id": embedder.model_id,
            "embedding_revision": embedder.revision,
            "embedding_model_class": getattr(embedder, "model_class", None),
            "embedding_pooling_method": getattr(embedder, "pooling_method", None),
            "normalization": "l2",
            "chunk_count": len(chunks),
            "dimensions": vectors.shape[1],
            "vectors_sha256": hashlib.sha256(vector_data).hexdigest(),
            "chunks_sha256": hashlib.sha256(chunk_data).hexdigest(),
        }
        _write_atomic(out_dir / "dense_vectors.npy", vector_data)
        _write_atomic(out_dir / "dense_chunks.jsonl", chunk_data)
        _write_atomic(
            out_dir / "dense_manifest.json",
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
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
        if (manifest.get("embedding_model_class"), manifest.get("embedding_pooling_method")) != (
            getattr(embedder, "model_class", None), getattr(embedder, "pooling_method", None)
        ):
            raise ValueError("embedding 配置与索引不一致")
        chunk_data = (out_dir / "dense_chunks.jsonl").read_bytes()
        vector_data = (out_dir / "dense_vectors.npy").read_bytes()
        if (hashlib.sha256(chunk_data).hexdigest() != manifest.get("chunks_sha256") or
                hashlib.sha256(vector_data).hexdigest() != manifest.get("vectors_sha256")):
            raise ValueError("向量索引文件与构建清单不一致")
        chunks = [Chunk(**json.loads(line)) for line in chunk_data.decode("utf-8").splitlines() if line]
        vectors = np.load(io.BytesIO(vector_data), allow_pickle=False)
        if len(chunks) != manifest["chunk_count"] or vectors.shape != (
            manifest["chunk_count"], manifest["dimensions"]
        ):
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
