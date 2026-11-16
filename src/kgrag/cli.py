"""项目命令行入口；当前提供完全离线的 ingest 子命令。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .ingest import build_corpus, load_sources


def main(argv: list[str] | None = None) -> int:
    """读取参数，校验本地文件并输出片段数量。"""
    parser = argparse.ArgumentParser(prog="kgrag")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="离线校验来源并构建片段")
    ingest.add_argument("--sources", type=Path, default=Path("config/sources.yaml"))
    ingest.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ingest.add_argument("--out-dir", type=Path, default=Path("data/index"))
    ingest.add_argument("--max-chars", type=int, default=1200)
    ingest.add_argument("--overlap-chars", type=int, default=120)
    args = parser.parse_args(argv)

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
