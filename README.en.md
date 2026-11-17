# Bilingual Graph RAG for Deep Learning Concepts

[简体中文](README.md) | [English](README.en.md)

This repository is being built as a local graph-enhanced RAG experiment using Chinese and English Markdown from *Dive into Deep Learning*. It will compare dense retrieval, unweighted one-hop graph expansion, and relation-aware multi-hop retrieval, with source passages grounding answers in Chinese.

The [implementation plan](docs/图谱增强RAG实施计划.md), [design](docs/图谱增强RAG项目设计草案.md), and [bilingual design supplement](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md) describe the research question, data workflow, and acceptance criteria.

The repository now contains project documents, original bilingual samples, and code for pinned textbook downloads and offline Markdown ingestion. The `fetch` command downloads and verifies the textbook files in the source manifest. The `ingest` command verifies local files offline, chunks them by section, and writes `data/index/chunks.jsonl` plus a local build manifest. Vector indexing, graph construction, retrieval, reranking, generation, and evaluation are still pending. Third-party textbook files and generated chunks stay local and are not committed.

## Download and local ingestion

The original sample files are already included. Run these commands to download the listed Chinese and English textbook files, then ingest them offline:

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` reuses locally verified files without network access and stops if an existing file or a download fails SHA256 verification. `ingest` does not access the network and also stops if a file is missing or its hash does not match. Character counts are the default chunking budget; `build_corpus` also accepts a token counter for the selected model. See the [build issue log](docs/RAG构建问题与解决方案.md) for problems and resolutions.

The project's original code is licensed under [MIT](LICENSE). That license does not cover the *Dive into Deep Learning* textbook content; consult the relevant versions of the [Chinese repository](https://github.com/d2l-ai/d2l-zh) and [English repository](https://github.com/d2l-ai/d2l-en) before using that content.

Maintenance rule: whenever this page changes, update the [Chinese README](README.md) in the same commit so both versions agree on facts, project status, and commands.
