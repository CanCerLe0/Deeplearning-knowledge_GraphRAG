# Bilingual Graph RAG for Deep Learning Concepts

[简体中文](README.md) | [English](README.en.md)

This repository is being built as a local graph-enhanced RAG experiment using Chinese and English Markdown from *Dive into Deep Learning*. It will compare dense retrieval, unweighted one-hop graph expansion, and relation-aware multi-hop retrieval, with source passages grounding answers in Chinese.

The [implementation plan](docs/图谱增强RAG实施计划.md), [design](docs/图谱增强RAG项目设计草案.md), and [bilingual design supplement](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md) describe the research question, data workflow, and acceptance criteria.

The repository now contains project documents, original bilingual samples, pinned textbook downloads, offline Markdown ingestion, a unified bilingual vector index, and a source-grounded Chinese answer baseline. `fetch` downloads and verifies listed files; `ingest` chunks them offline; `build-index` creates a local NumPy cosine index with a local embedding model; `ask` retrieves source passages and uses a configurable DeepSeek-compatible API to generate a Chinese answer. Graph construction, graph expansion, relation-aware reranking, and evaluation are still pending. Third-party textbook files and generated indexes stay local and are not committed.

## Download and local ingestion

The original sample files are already included. Run these commands to download the listed Chinese and English textbook files, then ingest them offline:

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` reuses locally verified files without network access and stops if an existing file or a download fails SHA256 verification. `ingest` does not access the network and also stops if a file is missing or its hash does not match. Character counts are the default chunking budget; `build_corpus` also accepts a token counter for the selected model. See the [build issue log](docs/RAG构建问题与解决方案.md) for problems and resolutions.

## Vector index and Chinese answers

Install the optional local embedding dependencies before building the index. The default model is `BAAI/bge-m3`. Its first use may download model files; the index records the resolved model revision. You can point `--embedding-model` to a local model directory, and should pin `--embedding-revision` for a formal experiment. Answer generation requires `DEEPSEEK_API_KEY`; index building does not.

```bash
python -m pip install -e '.[embedding]'
python -m kgrag.cli build-index --chunks data/index/chunks.jsonl
export DEEPSEEK_API_KEY=your_key_here
python -m kgrag.cli ask "注意力机制如何使用查询和键？"
```

`ask` outputs a Chinese answer, valid passage IDs, evidence languages and source links, plus model and generation settings. It returns an insufficient-evidence response when there is no evidence or no valid citation. If model memory is limited, select a smaller multilingual FlagEmbedding model with `--embedding-model` and rebuild the index with that same model; the three planned retrieval modes must share this index.

The project's original code is licensed under [MIT](LICENSE). That license does not cover the *Dive into Deep Learning* textbook content; consult the relevant versions of the [Chinese repository](https://github.com/d2l-ai/d2l-zh) and [English repository](https://github.com/d2l-ai/d2l-en) before using that content.

Maintenance rule: whenever this page changes, update the [Chinese README](README.md) in the same commit so both versions agree on facts, project status, and commands.
