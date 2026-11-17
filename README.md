# 面向深度学习概念的双语图谱增强 RAG

[简体中文](README.md) | [English](README.en.md)

本仓库正在构建一个基于《动手学深度学习》中英文 Markdown 的本地图谱增强 RAG 实验项目。项目将比较纯向量检索、无权一跳图扩展和关系感知的多跳检索，并用原文片段支持中文回答。

项目的[实施计划](docs/图谱增强RAG实施计划.md)、[设计草案](docs/图谱增强RAG项目设计草案.md)和[双语设计补充](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md)说明了研究问题、数据流程与验收标准。

目前仓库提供项目文档、自写中英文示例文本、固定版本教材下载、离线 Markdown 入库、统一双语向量索引和基于原文证据的中文基线回答。`fetch` 按来源清单下载并校验文件；`ingest` 离线分片；`build-index` 使用本地 embedding 构建 NumPy 余弦索引；`ask` 检索原文片段并调用可配置的 DeepSeek 兼容接口生成中文回答。图谱构建、图扩展、关系重排和评测仍待实现。第三方教材文件与生成的索引仅保存在本地，不提交到此仓库。

## 下载与本地入库

自写示例文件已包含在仓库中。运行以下命令下载清单中的中英文教材文件，再离线入库：

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` 会复用哈希校验通过的本地文件，此时无需联网；现有文件或下载内容的 SHA256 不匹配时会中止。`ingest` 不访问网络，文件缺失或哈希不符时也会中止。默认以字符数分片；`build_corpus` 也支持传入实际模型的 token 计数器。问题和处理过程见[构建问题记录](docs/RAG构建问题与解决方案.md)。

## 向量索引与中文问答

安装可选的本地 embedding 依赖后构建索引；默认模型为 `BAAI/bge-m3`，批量大小为 8，片段最大长度为 2048 token。可用 `--batch-size` 和 `--passage-max-length` 调整，并在 `build-index` 与 `ask` 中使用相同设置；索引会记录这些设置与实际模型版本。模型首次使用时可能需要下载。若已下载模型，也可用 `--embedding-model` 指向本地目录；正式实验建议用 `--embedding-revision` 固定版本。可用 `--device mps` 在支持的 Mac 上运行本地 embedding。生成答案默认使用 DeepSeek `deepseek-flash` 的非思考模式。首次在项目根目录复制密钥模板并限制权限，再用本地编辑器在 `.env` 的 `DEEPSEEK_API_KEY=` 后填写真实密钥：

```bash
cp .env.example .env
chmod 600 .env
```

此后从项目根目录运行，无须重复输入密钥；`build-index` 不需要密钥：

```bash
python -m pip install -e '.[embedding]'
python -m kgrag.cli build-index --chunks data/index/chunks.jsonl
python -m kgrag.cli ask "注意力机制如何使用查询和键？"
```

`ask` 自动读取项目根目录的 `.env`，已有环境变量优先；`.env.example` 仅保留空字段，不要填写真实密钥。`.gitignore` 已排除 `.env`。提交前用 `git status` 检查暂存文件，切勿强制添加密钥文件；若密钥已被推送，应立即在服务商处撤销并更换。

`ask` 输出逐项引用片段 ID 的中文回答、证据语言与原文链接，以及模型和生成参数。程序校验引用 ID 和引用标记；引用是否真正支持论断仍需人工核查。无证据或无有效引用时返回“证据不足”。模型显存不足时可用 `--embedding-model` 选择较小的多语言 FlagEmbedding 模型，同时用 `--embedding-class` 指定模型类型，必要时用 `--pooling-method` 指定池化方式，并用同一配置重建索引；三种计划中的检索方案须共用该索引。

本项目自写代码使用 [MIT 许可](LICENSE)。该许可不覆盖《动手学深度学习》教材正文；使用教材内容前请查看[中文仓库](https://github.com/d2l-ai/d2l-zh)和[英文仓库](https://github.com/d2l-ai/d2l-en)对应版本的许可说明。

维护约定：更新本页时，应在同一次提交中同步更新 [English README](README.en.md)，保持两种语言的事实、状态和命令一致。
