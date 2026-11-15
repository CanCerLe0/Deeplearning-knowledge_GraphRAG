# Bilingual Graph RAG for Deep Learning Concepts

[简体中文](README.md) | [English](README.en.md)

This repository is being built as a local graph-enhanced RAG experiment using Chinese and English Markdown from *Dive into Deep Learning*. It will compare dense retrieval, unweighted one-hop graph expansion, and relation-aware multi-hop retrieval, with source passages grounding answers in Chinese.

The [implementation plan](docs/图谱增强RAG实施计划.md), [design](docs/图谱增强RAG项目设计草案.md), and [bilingual design supplement](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md) describe the research question, data workflow, and acceptance criteria.

The repository currently contains project documents, Python package scaffolding, and original bilingual sample text. Textbook download, chunking, indexing, graph construction, retrieval, reranking, generation, and evaluation are not yet implemented. Third-party textbook files and generated indexes will remain local and will not be committed to this repository.

The project's original code is licensed under [MIT](LICENSE). That license does not cover the *Dive into Deep Learning* textbook content; consult the relevant versions of the [Chinese repository](https://github.com/d2l-ai/d2l-zh) and [English repository](https://github.com/d2l-ai/d2l-en) before using that content.

Maintenance rule: whenever this page changes, update the [Chinese README](README.md) in the same commit so both versions agree on facts, project status, and commands.
