# 第一版语料来源

本项目使用《动手学深度学习》两个上游仓库的 Markdown 原文，在本地构建检索语料。具体文件、标题、语言与逐文件 SHA256 记录在 [`config/sources.yaml`](../config/sources.yaml)。本地下载结果位于 `data/raw/`，该目录被 Git 忽略，不随本仓库发布。

| 语言 | 上游仓库 | 固定提交 | 文件数 | 本地目录 |
|---|---|---|---:|---|
| 中文 | [`d2l-ai/d2l-zh`](https://github.com/d2l-ai/d2l-zh) | `e6b18ccea71451a55fcd861d7b96fddf2587b09a` | 14 | `data/raw/d2l-zh/` |
| 英文 | [`d2l-ai/d2l-en`](https://github.com/d2l-ai/d2l-en) | `23d7a5aecceee57d1292c56e90cce307f183bb0a` | 14 | `data/raw/d2l-en/` |

选取范围为序列模型、编码器—解码器与序列到序列、注意力机制及 Transformer 的相关小节。中文和英文按主题选取，不假定文件或段落严格一一对应。中文仓库的 `*_origin.md` 翻译原稿副本未纳入，避免重复入库。

教材正文、由其构建的全文索引和缓存只保留在本地。项目代码的 MIT 许可不覆盖教材文本；复用或再分发前须核查上述固定提交中各仓库的许可与署名要求。
