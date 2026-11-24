# 面向深度学习概念的双语图谱增强 RAG

[简体中文](README.md) | [English](README.en.md)

本仓库正在构建一个基于《动手学深度学习》中英文 Markdown 的本地图谱增强 RAG 实验项目。已实现纯向量检索、无权一跳图扩展和关系感知的最多两跳检索，并用原文片段支持中文回答；已有 12 道开发题和三路初评命令，正式封存测试仍待完成。

项目的[实施计划](docs/图谱增强RAG实施计划.md)、[设计草案](docs/图谱增强RAG项目设计草案.md)和[双语设计补充](docs/双语本地Markdown设计补充.md)说明了研究问题、数据流程与验收标准。

目前仓库提供项目文档、自写中英文示例文本、固定版本教材下载、离线 Markdown 入库、统一双语向量索引、可审核的概念图，以及三路检索与原文证据支持的中文回答。`fetch` 按来源清单下载并校验文件；`ingest` 离线分片；`build-index` 使用本地 embedding 构建 NumPy 余弦索引；`build-graph` 调用可配置的 DeepSeek 兼容接口抽取关系并复用本地缓存；`ask` 按所选模式本地检索，再调用该接口生成中文回答；`eval` 对 12 道开发题运行三路对照。封存测试、正式评测和网页演示仍待实现。第三方教材文件与生成的索引仅保存在本地，不提交到此仓库。

## RAG 检索流程与当前策略

1. **Embedding：** 本地使用 `BAAI/bge-m3`（默认模型）将中文问题和中英文原文片段编码为同一空间的稠密向量；建索引时对片段向量做 L2 归一化，查询时也对问题向量做 L2 归一化。
2. **召回：** 用 NumPy 计算问题向量与索引中各片段向量的点积，即归一化后的余弦相似度，并按分数排序。`dense` 使用向量结果；`one_hop` 和 `typed_path` 从向量排名前 8 的片段出发，将片段正文中出现的图概念作为种子（即使该片段未承载图边证据），分别沿带原文证据的概念图扩展最多一跳和两跳候选。种子数可在 `config/retrieval.yaml` 中调整。
3. **重排与证据选择：** 当前不使用 cross-encoder 模型。`dense` 按归一化向量分排序；`one_hop` 在该分数上加入一跳奖励；`typed_path` 加权组合归一化向量分、最佳图路径分和问题概念覆盖度。路径分考虑关系类型匹配、边置信度与跳数惩罚。最终按相同的近似消息 token 预算、同一文档内的章节上限和双语近重复限制选择原文片段，供中文答案生成使用。

## 下载与本地入库

自写示例文件已包含在仓库中。运行以下命令下载清单中的中英文教材文件，再离线入库：

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` 会复用哈希校验通过的本地文件，此时无需联网；现有文件或下载内容的 SHA256 不匹配时会中止。`ingest` 不访问网络，文件缺失或哈希不符时也会中止。默认以字符数分片；`build_corpus` 也支持传入实际模型的 token 计数器。问题和处理过程见[构建问题记录](docs/RAG构建问题与解决方案.md)。

## 向量索引、概念图与中文问答

安装可选的本地 embedding 依赖后构建索引；默认模型为 `BAAI/bge-m3`，批量大小为 8，片段最大长度为 2048 token。可用 `--batch-size` 和 `--passage-max-length` 调整，并在 `build-index` 与 `ask` 中使用相同设置；索引会记录这些设置与实际模型版本。模型首次使用时可能需要下载。若已下载模型，也可用 `--embedding-model` 指向本地目录；正式实验建议用 `--embedding-revision` 固定版本。可用 `--device mps` 在支持的 Mac 上运行本地 embedding。生成答案默认使用 DeepSeek `deepseek-flash` 的非思考模式。首次在项目根目录复制密钥模板并限制权限，再用本地编辑器在 `.env` 的 `DEEPSEEK_API_KEY=` 后填写真实密钥：

```bash
cp .env.example .env
chmod 600 .env
```

此后从项目根目录运行，无须重复输入密钥；`build-index` 不需要密钥：

```bash
python -m pip install -e '.[embedding]'
python -m kgrag.cli build-index --chunks data/index/chunks.jsonl
python -m kgrag.cli build-graph --max-output-tokens 2048
python -m kgrag.cli ask "注意力机制如何使用查询和键？"
python -m kgrag.cli ask "束搜索与贪心搜索有什么区别？" --mode typed_path --token-budget 3000
```

`ask` 自动读取项目根目录的 `.env`，已有环境变量优先；`.env.example` 仅保留空字段，不要填写真实密钥。`.gitignore` 已排除 `.env`。提交前用 `git status` 检查暂存文件，切勿强制添加密钥文件；若密钥已被推送，应立即在服务商处撤销并更换。

`ask` 支持 `dense`（默认）、`one_hop` 和 `typed_path`，三者共用同一向量索引。输出包含证据语言、原文链接、概念路径、逐跳关系、证据 ID 和 `path_directions`（`forward`/`reverse` 表示相对原始有向关系的遍历方向），以及逐项引用片段 ID 的中文回答。`config/retrieval.yaml` 保存路径权重等开发期参数；`--token-budget` 对三种模式采用相同的中英混合近似计数，计入实际发送的系统提示词、问题、证据正文与元数据。正式评测前仍须接入所选生成模型的计数器，计入模型特定的消息开销。程序校验引用 ID 和引用标记；引用是否真正支持论断仍需人工核查。无证据或无有效引用时返回“证据不足”。图边抽查记录见[图谱抽取抽查](docs/图谱抽取抽查.md)。模型显存不足时可用 `--embedding-model` 选择较小的多语言 FlagEmbedding 模型，同时用 `--embedding-class` 指定模型类型，必要时用 `--pooling-method` 指定池化方式，并用同一配置重建索引。

## 开发题初评与扩展入库

`data/eval/dev.jsonl` 有 12 道按当前固定语料与默认分片配置核查的开发题：单概念、多跳、比较和不可回答题。每个必需证据组列出可以支持同一论点的候选片段 ID；新增语料或更改分片参数后，应重新核查这些 ID 与不可回答题边界。三路初评共用索引、检索配置、Top-k、上下文预算与生成参数：

```bash
python -m kgrag.cli eval --retrieval-only --questions data/eval/dev.jsonl --output data/runs/dev_retrieval.jsonl
python -m kgrag.cli eval --questions data/eval/dev.jsonl --output data/runs/dev_eval.jsonl
```

`--retrieval-only` 不创建生成客户端，输出不含教材正文，`answer` 为 `null`；完整评测输出逐题记录模式、片段 ID、语言、固定版本链接、答案、引用、检索及生成耗时、API token 用量和运行配置。两种输出均包含题集、向量索引和图审核状态的指纹。`data/runs/` 是本地输出，不提交；开发题只用于调试和初评，不代表封存测试集的最终效果。检索与回答初评见[开发集结果](docs/results.md)，正式封存测试与盲评仍待完成。检索初评需先完成下载、入库、索引和构图；完整评测还需设置密钥。

新增自写 Markdown 时，把文件放入 `data/sample/`，再在 `config/sources.yaml` 添加 `kind: local_markdown`、唯一 `source_id`、相对仓库根目录的 `path`、`title`、`language` 和 `license_note`；随后重新运行 `ingest`、`build-index` 和 `build-graph`。未改变的旧片段 ID 和原链接保持稳定。`fetch` 对校验通过的已下载教材文件可离线复用，`ingest` 与 `build-index` 本身不调用生成 API。

本项目自写代码使用 [MIT 许可](LICENSE)。该许可不覆盖《动手学深度学习》教材正文；使用教材内容前请查看[中文仓库](https://github.com/d2l-ai/d2l-zh)和[英文仓库](https://github.com/d2l-ai/d2l-en)对应版本的许可说明。

维护约定：更新本页时，应在同一次提交中同步更新 [English README](README.en.md)，保持两种语言的事实、状态和命令一致。
