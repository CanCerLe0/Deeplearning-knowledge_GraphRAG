"""项目命令行入口；下载与离线入库分开执行。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .fetch import download_url, fetch_sources
from .ingest import build_corpus, load_sources


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
    args = parser.parse_args(argv)

    if args.command == "fetch":
        sources = load_sources(args.sources)
        paths = fetch_sources(sources, args.raw_dir, download_url)
        print(f"已校验 {len(paths)} 个教材文件，保存于 {args.raw_dir}")
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
