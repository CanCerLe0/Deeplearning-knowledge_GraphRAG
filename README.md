# 面向深度学习概念的双语图谱增强 RAG

[简体中文](README.md) | [English](README.en.md)

本仓库正在构建一个基于《动手学深度学习》中英文 Markdown 的本地图谱增强 RAG 实验项目。项目将比较纯向量检索、无权一跳图扩展和关系感知的多跳检索，并用原文片段支持中文回答。

项目的[实施计划](docs/图谱增强RAG实施计划.md)、[设计草案](docs/图谱增强RAG项目设计草案.md)和[双语设计补充](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md)说明了研究问题、数据流程与验收标准。

目前仓库提供项目文档、自写中英文示例文本，以及固定版本教材下载和离线 Markdown 入库代码。`fetch` 命令按来源清单下载并校验教材文件；`ingest` 命令离线校验文件、按小节分片，并在本地生成 `data/index/chunks.jsonl` 和构建清单。向量索引、图谱构建、召回、重排、生成和评测仍待实现。第三方教材文件与生成的片段仅保存在本地，不提交到此仓库。

## 下载与本地入库

自写示例文件已包含在仓库中。运行以下命令下载清单中的中英文教材文件，再离线入库：

```bash
python -m pip install -e .
python -m kgrag.cli fetch --sources config/sources.yaml
python -m kgrag.cli ingest --sources config/sources.yaml
```

`fetch` 会复用哈希校验通过的本地文件，此时无需联网；现有文件或下载内容的 SHA256 不匹配时会中止。`ingest` 不访问网络，文件缺失或哈希不符时也会中止。默认以字符数分片；`build_corpus` 也支持传入实际模型的 token 计数器。问题和处理过程见[构建问题记录](docs/RAG构建问题与解决方案.md)。

本项目自写代码使用 [MIT 许可](LICENSE)。该许可不覆盖《动手学深度学习》教材正文；使用教材内容前请查看[中文仓库](https://github.com/d2l-ai/d2l-zh)和[英文仓库](https://github.com/d2l-ai/d2l-en)对应版本的许可说明。

维护约定：更新本页时，应在同一次提交中同步更新 [English README](README.en.md)，保持两种语言的事实、状态和命令一致。
