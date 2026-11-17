# 图谱增强 RAG 知识库 Implementation Plan

**Goal:** 在两周、每天 3–4 小时内，以本地中英文 Markdown 构建可扩展的深度学习知识库，验证关系感知的多跳图谱检索，并发布到作者个人 GitHub 公开仓库。

**Architecture:** 来源清单固定 `d2l-zh`、`d2l-en` 的提交和文件哈希，先下载中英文 Markdown 到本地，再离线解析为带稳定 ID 和语言元数据的片段；同一批片段同时进入本地统一双语向量索引和带原文证据的概念图。本地完成召回和重排，外接大模型承担关系候选抽取与中文答案生成。三种检索器共用 embedding、生成模型、提示词和上下文预算；评测脚本冻结开发/测试题与运行配置，网页展示答案、原文语言、引用和实际检索路径。

**Tech Stack:** Python 3.11+；Pydantic、PyYAML、Markdown 解析器、NumPy、NetworkX、FlagEmbedding（优先验证本地 BGE-M3，显存不足时配置较小的多语言模型）、可配置的 DeepSeek 兼容 Chat Completions API、Streamlit、pytest。依赖在实现时锁定；索引为本地 JSON/NumPy 文件，无需数据库服务。

**Spec:** [图谱增强 RAG 项目设计草案](./图谱增强RAG项目设计草案.md) 与 [双语本地 Markdown 设计补充](./双语本地Markdown设计补充.md)。两份设计与本计划已位于仓库根目录下的 `docs/`，随代码一同版本化。

## Global Constraints

- 问答与评测均使用中文；第一版语料是从《动手学深度学习》中英文仓库分别下载的序列模型、注意力机制与 Transformer 对应 Markdown；中文问题可引用英文证据。
- 两个仓库分别固定完整提交 SHA，来源清单逐文件记录 SHA256、语言、相对路径和许可备注；下载与入库分开，已校验文件支持离线重建，哈希不符则停止构建。
- 两周版 48 题，其中开发 12 题、封存测试 36 题；第一周只用开发题做初评。
- 纯向量、无权一跳、关系感知最多两跳三个方案须共用语料、chunking、embedding、生成模型、提示词和最终上下文 token 上限。
- 三种方案共用一个本地多语言 embedding 与统一双语向量索引；召回、图扩展和重排在本地执行，外接大模型仅处理关系候选抽取与基于证据的生成。
- 每条图边与回答引用都须追溯到原文片段 ID；图边不能单独充当答案事实来源。
- 新增本地 Markdown 来源只改来源清单并重建索引，不改检索或问答代码；第一版允许全量重建。
- 公开仓库仅包含自写双语示例文本、来源清单与下载/构建脚本；不提交 `data/raw/` 下的第三方教材 Markdown、全文索引、API 密钥、缓存或含密钥日志。
- 代码默认 MIT 许可；第三方资料单独标注来源与对应版本许可。正式发布前核实语料版本和来源条款。
- 默认 DeepSeek 客户端使用环境变量 `DEEPSEEK_API_KEY`；仓库只提交 `.env.example`，任何测试不读取真实密钥；生成客户端可替换。

## Review Focus

1. 上游下载内容与固定 SHA256 不符：中止获取和入库，不用旧缓存掩盖错误；Task 1 的伪下载器测试覆盖。
2. 同一文档重复入库或新增文档后旧引用失效：重建后 ID、记录数和旧 URL 稳定；Task 1 的幂等与扩展测试覆盖。
3. 中文问题无法召回英文证据，或无依据地合并中英文概念：Task 2 的跨语言召回测试与 Task 3 的别名校验测试覆盖。
4. API 返回不合法关系或不存在的证据 ID：该边被拒绝，构建继续并计数；Task 3 的校验测试覆盖。
5. 图扩展或双语近重复内容挤出相关证据：所有检索方案按相同 token 预算截断，按片段 ID 去重并限制同主题冗余；Task 4 的预算测试覆盖。
6. 英文语料使原本的不可回答题变为可回答：Task 6 在封存前重审金标准和拒答标签；Task 5 的无证据拒答测试覆盖。

---

## File Map

以包含 `docs/` 的当前项目文件夹作为仓库根目录；下列路径均相对该根目录。每个文件只承担一个职责。

| 路径 | 职责 |
|---|---|
| `src/kgrag/schema.py` | `SourceSpec`、`Document`、`Chunk`、`Edge`、`Evidence`、`Answer` 数据模型 |
| `src/kgrag/fetch.py` | 按固定提交下载中英文 Markdown、校验 SHA256、复用本地文件 |
| `src/kgrag/ingest.py` | 本地 Markdown 解析、按小节分块、稳定 ID、索引语料输出 |
| `src/kgrag/embed.py` | 本地 embedding 适配器与向量索引 |
| `src/kgrag/llm.py` | 可配置外接大模型客户端、DeepSeek 兼容适配器与假客户端 |
| `src/kgrag/graph.py` | 关系抽取、来源校验、别名归一、图保存与读取 |
| `src/kgrag/retrieve.py` | 三种检索模式、路径评分、上下文预算 |
| `src/kgrag/answer.py` | 带引用生成、引用校验、证据不足时拒答 |
| `src/kgrag/evaluate.py` | 题目格式、检索指标、实验运行与结果导出 |
| `src/kgrag/cli.py` | `fetch`、`ingest`、`build-graph`、`ask`、`evaluate` 命令 |
| `app.py` | Streamlit 演示，不含算法逻辑 |
| `config/sources.yaml` | 中英文仓库、完整提交 SHA、逐文件路径/语言/SHA256/许可及自写来源 |
| `data/raw/` | 下载的第三方 Markdown，仅保留本地并加入 `.gitignore` |
| `data/sample/` | 自写的小型双语 Markdown 文档；仅此示例语料进 Git |
| `data/eval/` | 自写题目与人工核查的中英文替代证据组，不含教材正文 |
| `tests/` | 无网络、无真实模型的单元和端到端小样本测试 |
| `docs/图谱增强RAG项目设计草案.md`、`docs/图谱增强RAG实施计划.md`、`docs/双语本地Markdown设计补充.md`、`docs/results.md`、`docs/data-sources.md` | 设计、实施计划、补充设计、实验与来源 |

## Shared Interfaces

- `SourceSpec(source_id: str, kind: Literal["github_markdown", "local_markdown"], path: str, title: str, language: Literal["zh", "en"], license_note: str, repo: str | None = None, revision: str | None = None, sha256: str | None = None)`；GitHub 来源须有仓库、完整提交 SHA 和文件 SHA256，`path` 相对上游仓库根目录；本地自写来源可不填后三项，`path` 相对本项目根目录。GitHub 文件落在 `data/raw/<仓库名>/<path>`。
- `Chunk(chunk_id: str, source_id: str, document_id: str, section: str, text: str, language: Literal["zh", "en"], source_path: str, line_start: int | None, line_end: int | None, url: str | None)`；片段 ID 基于来源、上游路径或本地路径、小节锚点和规范化文本哈希，与导入顺序无关；GitHub URL 固定到提交 SHA，并可指向行号。
- `Edge(subject: str, predicate: Literal["prerequisite_of", "component_of", "contrasts_with", "used_in"], object: str, evidence_chunk_id: str, confidence: float)`。
- `Evidence(chunk: Chunk, score: float, path: tuple[str, ...])`；三种检索模式统一返回此类型。
- `Answer(text: str, citation_chunk_ids: list[str])`。
- `Embedder.encode(texts: Sequence[str]) -> np.ndarray` 与 `ChatClient.complete_json(messages: list[dict]) -> dict` 均可替换为测试实现。

### Task 1: 仓库骨架与双语 Markdown 入库（第 1 天）

**Files:** `pyproject.toml`、`.gitignore`、`.env.example`、`config/sources.yaml`、`data/sample/attention_zh.md`、`data/sample/attention_en.md`、`src/kgrag/schema.py`、`src/kgrag/fetch.py`、`src/kgrag/ingest.py`、`src/kgrag/cli.py`、`tests/test_fetch.py`、`tests/test_ingest.py`、现有 `docs/` 设计与计划。

**Interfaces:** 产出 `load_sources(path: Path) -> list[SourceSpec]`、`fetch_sources(specs: list[SourceSpec], raw_dir: Path, downloader: Callable[[str], bytes]) -> list[Path]`、`build_corpus(specs: list[SourceSpec], raw_dir: Path, out_dir: Path) -> list[Chunk]`；CLI 的 `fetch` 提供网络下载器，`ingest` 只读取已校验的本地文件；后续任务读取 `out_dir/chunks.jsonl`。

- [ ] **Step 1: 写失败测试。** 用伪下载器和自写中英文 Markdown 断言：只下载清单列出的文件、校验两个仓库各自的完整提交 SHA 与文件 SHA256、校验失败中止、已校验文件可离线复用；`*_origin.md` 不入库。再断言新增本地来源后旧 `chunk_id` 与固定版本 URL 不变，重复 `build_corpus` 的记录和 ID 完全相同。解析测试需覆盖首个标题之前的 D2L 执行代码块，以及标题、公式、普通代码块；保留来源位置并去除构建指令。
- [ ] **Step 2: 运行 `python -m pytest tests/test_fetch.py tests/test_ingest.py -v`。** 预期因接口不存在而失败。
- [ ] **Step 3: 实现来源清单、固定提交下载与哈希校验、本地 Markdown 解析和稳定分块。** 下载的教材文件落在 `data/raw/<仓库名>/<path>`，建立 `.gitignore` 排除 `data/raw/`、生成索引、模型、缓存和 `.env`；`ingest` 不访问网络或大模型。固定中英文两个仓库的实际提交和逐文件路径/哈希，来源列表只覆盖第一版主题。
- [ ] **Step 4: 再运行同一测试。** 预期全部通过；运行 `python -m kgrag.cli fetch --sources config/sources.yaml` 后运行 `python -m kgrag.cli ingest --sources config/sources.yaml`，确认两个仓库的片段都出现；断网或使用伪下载器验证入库可离线重建，再加入自写 Markdown 来源，确认新片段出现且旧 ID 不变。
- [ ] **Step 5: 提交。** `git add` 本任务文件，`git commit -m "feat: add extensible corpus ingestion"`。

### Task 2: 本地向量检索与基线回答（第 2 天）

**Files:** `src/kgrag/embed.py`、`src/kgrag/llm.py`、`src/kgrag/answer.py`、`src/kgrag/cli.py`、`tests/test_dense_answer.py`。

**Interfaces:** 产出 `DenseIndex.build(chunks: list[Chunk], embedder: Embedder, out_dir: Path) -> DenseIndex`、`DenseIndex.search(query: str, k: int) -> list[Evidence]`、`generate_answer(question: str, evidence: list[Evidence], client: ChatClient) -> Answer`。

- [ ] **Step 1: 写失败测试。** 自写中英文片段与假多语言 embedding 使中文问题召回英文证据并排第一，且结果保留 `language="en"` 和固定版本链接；空证据时返回中文“证据不足”且引用为空；生成器引用不在传入 evidence 内的 ID 时过滤该引用。
- [ ] **Step 2: 运行 `python -m pytest tests/test_dense_answer.py -v`。** 预期失败。
- [ ] **Step 3: 实现统一双语 NumPy 余弦检索、可配置本地多语言 embedding 与外接生成客户端。** 同一模型配置贯穿所有检索方案；记录 embedding/生成模型 ID、版本和生成参数。默认提供 DeepSeek 兼容适配器；回答提示词固定为中文回答，仅允许引用给定片段 ID 并显示证据语言。
- [ ] **Step 4: 测试通过后，用含英文证据的 3 个自写片段完成无密钥的假客户端端到端演示；如已有用户提供的密钥环境变量，单独做一次真实 API 冒烟检查。** 不打印密钥或完整敏感环境。
- [ ] **Step 5: 提交。** `git commit -m "feat: add dense retrieval and grounded answers"`。

### Task 3: 带来源的概念图（第 3–4 天）

**Files:** `src/kgrag/graph.py`、`src/kgrag/cli.py`、`tests/test_graph.py`、`docs/data-sources.md`。

**Interfaces:** 产出 `build_graph(chunks: list[Chunk], client: ChatClient, out_dir: Path) -> GraphIndex`；`GraphIndex.neighbors(concept: str, max_hops: int) -> list[GraphPath]`；`GraphPath` 含节点、关系和证据片段 ID。

- [ ] **Step 1: 写失败测试。** 模拟 API 返回两条有效关系、一条非法关系和一个不存在的 `evidence_chunk_id`；断言只保留有效边并记录拒绝数；经显式验证的中英文概念别名可跨两个片段形成两跳路径，未经验证的相似译名不得自动合并片段或证据 ID。
- [ ] **Step 2: 运行 `python -m pytest tests/test_graph.py -v`。** 预期失败。
- [ ] **Step 3: 实现结构化抽取、四类关系校验、证据 ID 校验、可审核的中英文别名映射、图持久化。** 图边必须可回溯带语言和版本的原文片段；按章节和语言批处理 API 请求并缓存结果，重复构建不重复计费。
- [ ] **Step 4: 测试通过；在下载的中英文教材小节上构图，抽查首批关系与跨语言别名并记录错误类型。** 实验图须记录两个仓库的来源提交、抽取模型和提示词版本。
- [ ] **Step 5: 提交。** `git commit -m "feat: build provenance-aware concept graph"`。

### Task 4: 三路检索与自定义路径评分（第 5–6 天）

**Files:** `src/kgrag/retrieve.py`、`src/kgrag/cli.py`、`tests/test_retrieve.py`。

**Interfaces:** 产出 `retrieve(question: str, mode: Literal["dense", "one_hop", "typed_path"], k: int, token_budget: int) -> list[Evidence]`。`typed_path` 分数为 `α·归一化向量相似度 + β·最佳路径分 + γ·问题概念覆盖度`；路径分包含关系类型匹配、边置信度与跳数惩罚。

- [ ] **Step 1: 写失败测试。** 中文问题通过经验证的跨语言概念路径让英文两跳证据在 `typed_path` 中进入 Top-k；`dense` 不访问图；`one_hop` 最多扩展一跳；所有模式使用相同 token 预算，按片段 ID 去重并限制同主题双语近重复片段，无证据片段的路径不能进入结果。
- [ ] **Step 2: 运行 `python -m pytest tests/test_retrieve.py -v`。** 预期失败。
- [ ] **Step 3: 实现基于统一双语索引的三种本地检索模式、关系意图匹配、路径评分及贪心证据选择。** `α/β/γ`、跳数惩罚和同主题双语冗余限制写入配置；仅用开发题调参。为消融提供 `β=0`、关闭类型权重两个开关。
- [ ] **Step 4: 测试通过；人工检查至少 3 道跨概念问题的路径与中英文原文证据，其中至少 1 道中文问题使用英文证据。** 对照相同语料、embedding、生成模型、提示词和上下文预算。
- [ ] **Step 5: 提交。** `git commit -m "feat: add typed multi-hop graph retrieval"`。

### Task 5: 第一周验收与入库扩展检查（第 7 天）

**Files:** `src/kgrag/evaluate.py`、`data/eval/dev.jsonl`、`tests/test_pipeline.py`、`README.md`。

**Interfaces:** 产出 `run_eval(questions_path: Path, modes: list[str], output_path: Path) -> None`，结果逐题记录模式、检索片段 ID、证据语言、固定版本 URL、答案、引用、延迟和 token 用量。

- [ ] **Step 1: 写失败端到端测试。** 用自写中英文小语料运行 `fetch → ingest → dense/graph → answer`，断言中文问题可引用英文原文并以中文作答；已有校验文件离线重建时不调用下载器或大模型。再增加一个本地 Markdown 文件重建，断言新问题可命中新片段且旧引用 ID 和 URL 保持稳定；无证据问题拒答。
- [ ] **Step 2: 运行 `python -m pytest tests/test_pipeline.py -v`。** 预期失败。
- [ ] **Step 3: 实现评测运行器和 12 道已人工核查的开发题。** 每题用必需证据组记录可替代的中英文片段 ID；README 写明双语下载、哈希校验、离线入库和问答命令。开发集结果标注为“调试/初评”，不用作最终效果结论。
- [ ] **Step 4: 运行完整测试与开发题三路对照。** 预期可从已下载且校验通过的中英文 Markdown 重建干净索引并输出同字段 JSONL；人工确认至少一个英文固定版本链接和一个新增本地来源的可定位引用。
- [ ] **Step 5: 提交。** `git commit -m "feat: add reproducible development evaluation"`。

### Task 6: 封存测试、指标与消融（第 8–11 天）

**Files:** `data/eval/test.jsonl`、`src/kgrag/evaluate.py`、`tests/test_metrics.py`、`docs/results.md`、`scripts/score_annotations.py`。

**Interfaces:** 产出 `recall_at_k(required_groups: list[set[str]], retrieved: list[str], k: int) -> float`、`complete_evidence_hit(required_groups: list[set[str]], retrieved: list[str], k: int) -> bool` 和每题人工评分模板；每组为同一论点的人工确认替代片段，命中其中一个即可。输出按题型及证据语言分组的结果表和失败案例。

- [ ] **Step 1: 写失败测试。** 构造 2 个必需证据组、只命中 1 组的例子，断言 `Recall@5=0.5` 且完整证据命中为 `False`；其中一组含可替代的中英文片段 ID，命中任一只计一次。空证据组题不进入证据召回平均值，但用于拒答评分。
- [ ] **Step 2: 运行 `python -m pytest tests/test_metrics.py -v`。** 预期失败。
- [ ] **Step 3: 实现证据组指标、盲评模板和消融配置。** 逐题核查中英文可替代证据并封存 36 道测试题；加入英文后重审不可回答题，至少一半题由作者依据原文自行撰写，测试题不从已抽取图边反向生成。抽查至少 40 条图边，记录 precision 与典型错误。
- [ ] **Step 4: 固定两个仓库版本、语料清单及检索参数后仅运行一次主测试集对照；对三种模式的 36 题答案做匿名人工评分。** 记录证据组 Recall@5、多跳完整召回、答案/引用正确率、拒答准确率、证据语言、延迟与 API 用量；做 `β=0` 与关闭类型权重的消融，并在 `docs/results.md` 写明样本规模的限制。
- [ ] **Step 5: 测试通过并提交。** `git commit -m "eval: add held-out comparison and ablations"`。

### Task 7: 演示与公开仓库材料（第 12–13 天）

**Files:** `app.py`、`README.md`、`.env.example`、`LICENSE`、`docs/results.md`、`docs/data-sources.md`、`tests/test_release_fixture.py`。

**Interfaces:** UI 调用已有 `retrieve` 与 `generate_answer`，展示三路答案、引用原文、被使用的局部概念路径；不在 UI 中复制检索算法。

- [ ] **Step 1: 写失败的发布样例测试。** 无 API 密钥时，使用自写样例语料和假客户端执行一条问答；断言 README 所列本地命令可运行，新增 Markdown 来源可查到。
- [ ] **Step 2: 运行 `python -m pytest tests/test_release_fixture.py -v`。** 预期失败。
- [ ] **Step 3: 实现 Streamlit 页面、预设问题、引用语言与局部路径视图；补齐 README 首屏的研究问题、方法图、对照表、消融和失败分析。** 写清两个仓库的下载、校验、作者、提交与许可；源码仓库只含自写示例，第三方教材由复现者本地下载。
- [ ] **Step 4: 运行测试和 `streamlit run app.py`；手动检查一个引用英文原文的成功案例、一个失败案例及新增来源。** 截图或录制 1–2 分钟演示。
- [ ] **Step 5: 提交。** `git commit -m "docs: add graph RAG demo and reproducibility guide"`。

### Task 8: 干净环境复现与 GitHub 发布（第 14 天）

**Files:** 仅修复复现或发布检查发现的问题；更新 `README.md`、`docs/results.md` 与必要配置。

**Interfaces:** 最终交付个人账号下公开 GitHub 仓库 URL；代码、设计、实验结果和演示可从仓库访问。

- [ ] **Step 1: 在新虚拟环境运行 README 的安装、双语下载/校验、离线入库、测试、问答和评测命令。** 记录 Python 版本、依赖锁定、多语言 embedding 模型 ID、生成模型 ID、两个仓库的提交与文件哈希、运行配置。
- [ ] **Step 2: 检查 `git status`、待提交文件和 Git 历史。** 确认无 `.env`、真实密钥、`data/raw/` 下的中英文教材正文、全文索引或含密钥日志；运行全套 `python -m pytest -q`，有可用密钥时再做一次真实 API 冒烟检查。
- [ ] **Step 3: 修复具体问题并重复失败的检查。** README 中的运行结果与 `docs/results.md` 一致。
- [ ] **Step 4: 用作者自己的 GitHub 账号创建公开仓库并推送。** 先检查当前 GitHub 工具的认证状态；需要时连接作者账号。确认仓库所有者、可见性为 Public、默认分支包含全部交付物。
- [ ] **Step 5: 打开公开仓库 URL，核对 README、许可、结果图、演示和克隆后的复现命令。** 记录最终 URL 与提交 SHA，供简历和面试使用。

## Self-Review

- 设计稿的双语下载与校验、本地处理、跨语言证据、算法对照、消融、可扩展入库、出处追溯、公开发布和时间预算分别落在 Task 1–8。
- 所有测试使用自写 fixture/假客户端；真实 API 冒烟单独执行，不进入 CI。
- Task 1–5 形成第 7 天可运行版本；Task 6–8 形成第 14 天公开可复现版本。
- 哈希错误、离线重建、跨语言召回、新增来源、重复入库、错误图边、双语上下文挤占和无答案问题都有明确测试。
- 许可处理采用不随仓库再发布第三方正文的方式；正式复现仍需固定和记录上游来源版本。
