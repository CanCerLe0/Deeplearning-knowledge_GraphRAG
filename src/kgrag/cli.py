"""项目命令行入口；下载与离线入库分开执行。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .answer import generate_answer
from .embed import DenseIndex, FlagEmbedder, load_chunks
from .fetch import download_url, fetch_sources
from .ingest import build_corpus, load_sources
from .llm import DeepSeekChatClient


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
    ask = commands.add_parser("ask", help="向量检索并依据原文证据生成中文回答")
    ask.add_argument("question")
    ask.add_argument("--index-dir", type=Path, default=Path("data/index/dense"))
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--chat-model", default="deepseek-chat")
    ask.add_argument("--base-url", default="https://api.deepseek.com")
    ask.add_argument("--temperature", type=float, default=0.0)
    ask.add_argument("--max-output-tokens", type=int, default=1024)
    for command in (build_index, ask):
        command.add_argument("--embedding-model", default="BAAI/bge-m3")
        command.add_argument("--embedding-revision", default="main")
        command.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    if args.command == "fetch":
        sources = load_sources(args.sources)
        paths = fetch_sources(sources, args.raw_dir, download_url)
        print(f"已校验 {len(paths)} 个教材文件，保存于 {args.raw_dir}")
        return 0

    if args.command in ("build-index", "ask"):
        embedder = FlagEmbedder(
            model_id=args.embedding_model, revision=args.embedding_revision, device=args.device
        )
        if args.command == "build-index":
            chunks = load_chunks(args.chunks)
            index = DenseIndex.build(chunks, embedder, args.out_dir)
            print(f"已索引 {len(index.chunks)} 个片段，输出：{args.out_dir}")
            return 0

        index = DenseIndex.load(args.index_dir, embedder)
        evidence = index.search(args.question, args.top_k)
        client = DeepSeekChatClient(
            model=args.chat_model, base_url=args.base_url,
            temperature=args.temperature, max_tokens=args.max_output_tokens,
        )
        answer = generate_answer(args.question, evidence, client)
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
                "generation_model_id": client.model,
                "generation_model_version": client.last_model,
                "temperature": client.temperature,
                "max_output_tokens": client.max_tokens,
                "top_k": args.top_k,
            },
            "usage": client.last_usage,
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
