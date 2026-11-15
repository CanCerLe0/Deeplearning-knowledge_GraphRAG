# 面向深度学习概念的双语图谱增强 RAG

[简体中文](README.md) | [English](README.en.md)

本仓库正在构建一个基于《动手学深度学习》中英文 Markdown 的本地图谱增强 RAG 实验项目。项目将比较纯向量检索、无权一跳图扩展和关系感知的多跳检索，并用原文片段支持中文回答。

项目的[实施计划](docs/图谱增强RAG实施计划.md)、[设计草案](docs/图谱增强RAG项目设计草案.md)和[双语设计补充](docs/superpowers/specs/2026-11-15-bilingual-local-markdown-rag-design.md)说明了研究问题、数据流程与验收标准。

目前仓库包含项目文档、Python 工程骨架和自写的中英文示例文本。教材下载、分片、索引、图谱构建、召回、重排、生成和评测功能尚未实现。第三方教材文件与生成的索引将仅保存在本地，不提交到此仓库。

本项目自写代码使用 [MIT 许可](LICENSE)。该许可不覆盖《动手学深度学习》教材正文；使用教材内容前请查看[中文仓库](https://github.com/d2l-ai/d2l-zh)和[英文仓库](https://github.com/d2l-ai/d2l-en)对应版本的许可说明。

维护约定：更新本页时，应在同一次提交中同步更新 [English README](README.en.md)，保持两种语言的事实、状态和命令一致。
