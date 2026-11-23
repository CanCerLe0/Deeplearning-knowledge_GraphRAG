"""项目命令行入口；下载与离线入库分开执行。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from .answer import REFUSAL, generate_answer
from .embed import DenseIndex, FlagEmbedder, load_chunks
from .fetch import download_url, fetch_sources
from .graph import build_graph
from .ingest import build_corpus, load_sources
from .llm import DeepSeekChatClient
from .schema import Answer


def main(argv: list[str] | None = None) -> int:
    """按子命令下载来源文件或离线构建片段。"""
    parser = argparse.ArgumentParser(prog="kgrag")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="下载或校验清单中的固定版本 Markdown")
    fetch.add_argument("--sources", type=Path, default=Path("config/sources.yaml"))
    fetch.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ingest = commands.add_parser("ingest", help="离线校验来源并构建片段")
    ingest.add_argument("--sources", type=Path, default=Path("config/sources.yaml"))
    ingest.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ingest.add_argument("--out-dir", type=Path, default=Path("data/index"))
    ingest.add_argument("--max-chars", type=int, default=1200)
    ingest.add_argument("--overlap-chars", type=int, default=120)
    build_index = commands.add_parser("build-index", help="从入库片段构建双语向量索引")
    build_index.add_argument("--chunks", type=Path, default=Path("data/index/chunks.jsonl"))
    build_index.add_argument("--out-dir", type=Path, default=Path("data/index/dense"))
    build_graph_command = commands.add_parser("build-graph", help="抽取并校验有来源的概念关系")
    build_graph_command.add_argument("--chunks", type=Path, default=Path("data/index/chunks.jsonl"))
    build_graph_command.add_argument("--out-dir", type=Path, default=Path("data/index/graph"))
    build_graph_command.add_argument("--aliases", type=Path, default=Path("config/concept_aliases.yaml"))
    build_graph_command.add_argument("--chat-model", default="deepseek-flash")
    build_graph_command.add_argument("--base-url", default="https://api.deepseek.com")
    build_graph_command.add_argument("--temperature", type=float, default=0.0)
    build_graph_command.add_argument("--max-output-tokens", type=int, default=1024)
    build_graph_command.add_argument("--batch-size", type=int, default=8)
    ask = commands.add_parser("ask", help="向量检索并依据原文证据生成中文回答")
    ask.add_argument("question")
    ask.add_argument("--index-dir", type=Path, default=Path("data/index/dense"))
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--chat-model", default="deepseek-flash")
    ask.add_argument("--base-url", default="https://api.deepseek.com")
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--max-output-tokens", type=int, default=1024)
    for command in (build_index, ask):
        command.add_argument("--embedding-model", default="BAAI/bge-m3")
        command.add_argument("--embedding-revision", default="main")
        command.add_argument("--embedding-class", default=None)
        command.add_argument("--pooling-method", default=None)
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--passage-max-length", type=int, default=None)
        command.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.command == "fetch":
        sources = load_sources(args.sources)
        paths = fetch_sources(sources, args.raw_dir, download_url)
        print(f"已校验 {len(paths)} 个教材文件，保存于 {args.raw_dir}")
        return 0

    if args.command == "build-graph":
        chunks = load_chunks(args.chunks)
        alias_data = yaml.safe_load(args.aliases.read_text(encoding="utf-8"))
        if not isinstance(alias_data, dict) or not isinstance(alias_data.get("aliases"), list):
            raise ValueError("别名文件必须包含 aliases 列表")
        manifest_path = args.chunks.parent / "build_manifest.json"
        corpus_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else None
        client = DeepSeekChatClient(
            model=args.chat_model, base_url=args.base_url,
            temperature=args.temperature, max_tokens=args.max_output_tokens,
        )
        graph = build_graph(
            chunks, client, args.out_dir, aliases=alias_data["aliases"],
            batch_size=args.batch_size, corpus_manifest=corpus_manifest,
        )
        print(f"构图完成：有效关系 {len(graph.edges)}，拒绝 {sum(graph.rejected.values())}，输出：{args.out_dir}")
        return 0

    if args.command in ("build-index", "ask"):
        embedder = FlagEmbedder(
            model_id=args.embedding_model, revision=args.embedding_revision,
            model_class=args.embedding_class, pooling_method=args.pooling_method,
            batch_size=args.batch_size, passage_max_length=args.passage_max_length,
            device=args.device,
        )
        if args.command == "build-index":
            chunks = load_chunks(args.chunks)
            index = DenseIndex.build(chunks, embedder, args.out_dir)
            print(f"已索引 {len(index.chunks)} 个片段，输出：{args.out_dir}")
            return 0

        index = DenseIndex.load(args.index_dir, embedder)
        evidence = index.search(args.question, args.top_k)
        if evidence:
            client = DeepSeekChatClient(
                model=args.chat_model, base_url=args.base_url,
                temperature=args.temperature, max_tokens=args.max_output_tokens,
            )
            answer = generate_answer(args.question, evidence, client)
            generation_model_version = client.last_model
            usage = client.last_usage
        else:
            answer = Answer(REFUSAL, [])
            generation_model_version = None
            usage = {}
        result = {
            "question": args.question,
            "answer": {
                "text": answer.text,
                "citation_chunk_ids": answer.citation_chunk_ids,
            },
            "evidence": [
                {
                    "chunk_id": item.chunk.chunk_id,
                    "language": item.chunk.language,
                    "section": item.chunk.section,
                    "score": item.score,
                    "url": item.chunk.url,
                    "text": item.chunk.text,
                }
                for item in evidence
            ],
            "run_config": {
                "embedding_model_id": embedder.model_id,
                "embedding_revision": embedder.revision,
                "embedding_model_class": getattr(embedder, "model_class", None),
                "embedding_pooling_method": getattr(embedder, "pooling_method", None),
                "embedding_batch_size": getattr(embedder, "batch_size", None),
                "embedding_passage_max_length": getattr(embedder, "passage_max_length", None),
                "generation_model_id": args.chat_model,
                "generation_model_version": generation_model_version,
                "temperature": args.temperature,
                "max_output_tokens": args.max_output_tokens,
                "top_k": args.top_k,
            },
            "usage": usage,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    # 下载与入库分开：此命令只读本地 Markdown，不访问网络或大模型。
    sources = load_sources(args.sources)
    chunks = build_corpus(
        sources, args.raw_dir, args.out_dir,
        max_chars=args.max_chars, overlap_chars=args.overlap_chars,
    )
    counts = Counter(chunk.language for chunk in chunks)
    print(f"已处理 {len(sources)} 个来源，生成 {len(chunks)} 个片段")
    print(f"语言分布：{dict(counts)}")
    print(f"输出：{args.out_dir / 'chunks.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
