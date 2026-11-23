# Bilingual Graph RAG for Deep Learning Concepts

[简体中文](README.md) | [English](README.en.md)

This repository is being built as a local graph-enhanced RAG experiment using Chinese and English Markdown from *Dive into Deep Learning*. Dense retrieval, unweighted one-hop graph expansion, and relation-aware retrieval up to two hops are implemented, with source passages grounding answers in Chinese; formal comparative evaluation is still pending.

The [implementation plan](docs/图谱增强RAG实施计划.md), [design](docs/图谱增强RAG项目设计草案.md), and [bilingual design supplement](docs/双语本地Markdown设计补充.md) describe the research question, data workflow, and acceptance criteria.

The repository now contains project documents, original bilingual samples, pinned textbook downloads, offline Markdown ingestion, a unified bilingual vector index, a reviewable concept graph, and three retrieval modes with source-grounded Chinese answers. `fetch` downloads and verifies listed files; `ingest` chunks them offline; `build-index` creates a local NumPy cosine index with a local embedding model; `build-graph` extracts relations through a configurable DeepSeek-compatible API and reuses a local cache; `ask` retrieves locally in the selected mode, then uses that API to generate a Chinese answer. Development questions, formal evaluation, and the web demo are still pending. Third-party textbook files and generated indexes stay local and are not committed.

## RAG retrieval stages and current strategy

1. **Embedding:** The local `BAAI/bge-m3` model (the default) encodes Chinese questions and Chinese and English source passages as dense vectors in one shared space. Passage vectors are L2-normalized when the index is built, and question vectors are L2-normalized at query time.
2. **Recall:** NumPy computes the dot product between the question vector and every indexed passage vector, which equals cosine similarity after normalization, then ranks the passages. `dense` uses the vector results; `one_hop` and `typed_path` start from the top 8 vector-ranked passages and expand candidates by at most one and two hops, respectively, through the provenance-backed concept graph. The seed count is configurable in `config/retrieval.yaml`.
3. **Reranking and evidence selection:** No cross-encoder model is currently used. `dense` ranks by normalized vector score; `one_hop` adds a one-hop bonus; `typed_path` combines normalized vector score, the best graph-path score, and question-concept coverage with configurable weights. Path scores account for relation-type match, edge confidence, and a hop penalty. The final source passages are selected under the same context token budget, per-section limit, and bilingual near-duplicate limit before Chinese answer generation.

## Download and local ingestion

The original sample files are already included. Run these commands to download the listed Chinese and English textbook files, then ingest them offline:

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` reuses locally verified files without network access and stops if an existing file or a download fails SHA256 verification. `ingest` does not access the network and also stops if a file is missing or its hash does not match. Character counts are the default chunking budget; `build_corpus` also accepts a token counter for the selected model. See the [build issue log](docs/RAG构建问题与解决方案.md) for problems and resolutions.

## Vector index, concept graph, and Chinese answers

Install the optional local embedding dependencies before building the index. The default model is `BAAI/bge-m3`, with batch size 8 and a maximum passage length of 2048 tokens. Adjust these with `--batch-size` and `--passage-max-length`, using the same settings for `build-index` and `ask`; the index records them along with the resolved model revision. Its first use may download model files. You can point `--embedding-model` to a local model directory, and should pin `--embedding-revision` for a formal experiment. Use `--device mps` to run local embeddings on a supported Mac. Answer generation defaults to DeepSeek `deepseek-flash` in non-thinking mode. In the project root, copy the key template once, restrict its permissions, then use a local editor to fill in `DEEPSEEK_API_KEY=` in `.env`:

```bash
cp .env.example .env
chmod 600 .env
```

After that, run from the project root without entering the key again. `build-index` does not need a key:

```bash
python -m pip install -e '.[embedding]'
python -m kgrag.cli build-index --chunks data/index/chunks.jsonl
python -m kgrag.cli build-graph --max-output-tokens 2048
python -m kgrag.cli ask "注意力机制如何使用查询和键？"
python -m kgrag.cli ask "束搜索与贪心搜索有什么区别？" --mode typed_path --token-budget 3000
```

`ask` loads `.env` from the project root automatically, while an existing environment variable takes precedence. Keep `.env.example` empty of real secrets. `.gitignore` excludes `.env`; check staged files with `git status` before committing, and never force-add a key file. If a key was pushed, revoke and replace it with the provider immediately.

`ask` supports `dense` (default), `one_hop`, and `typed_path`, all using the same vector index. Its output includes evidence languages, source links, concept paths with per-hop relations and evidence IDs, and a Chinese answer citing passage IDs for individual claims. `config/retrieval.yaml` holds development-stage path weights and other parameters. `--token-budget` uses the same approximate mixed Chinese and English token count in all three modes; formal evaluation still needs the selected generation model's tokenizer. The program checks citation IDs and markers; whether a passage actually supports a claim still requires human review. It returns an insufficient-evidence response when there is no evidence or no valid citation. See the [graph extraction audit](docs/图谱抽取抽查.md). If model memory is limited, select a smaller multilingual FlagEmbedding model with `--embedding-model`, specify its model type with `--embedding-class` and, if needed, its pooling method with `--pooling-method`; rebuild the index with the same configuration.

The project's original code is licensed under [MIT](LICENSE). That license does not cover the *Dive into Deep Learning* textbook content; consult the relevant versions of the [Chinese repository](https://github.com/d2l-ai/d2l-zh) and [English repository](https://github.com/d2l-ai/d2l-en) before using that content.

Maintenance rule: whenever this page changes, update the [Chinese README](README.md) in the same commit so both versions agree on facts, project status, and commands.
