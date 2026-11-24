# 开发集初评（检索与回答）

2026-11-24 修订 `data/eval/dev.jsonl` 的证据组与不可回答题后，对固定版本双语索引重新运行三路初评。10 道可回答题共有 15 个必需证据组；另外 2 道领域内但语料缺少具体实验结果的题不计入证据召回分母。本页是调试记录，不是封存测试集结果；回答部分只统计程序输出，不作答案正确率或引用支持率结论。旧版题集和初评保留在 Git 历史中，不与本次数字混算。

| 检索模式 | 必需证据组命中 / 15 | 完整证据命中 / 10 题 | Top-5 含英文片段 / 12 题 |
| --- | ---: | ---: | ---: |
| `dense` | 15 | 10 | 5 |
| `one_hop` | 15 | 10 | 5 |
| `typed_path` | 11 | 6 | 10 |

这里的“命中”指 Top-5 中至少有一个片段 ID 落入人工核查的同一论点候选组。“完整证据命中”要求一道题的所有必需组均命中。英文片段计数仅表示进入上下文，不表示答案引用了英文。三种模式共用 555 个片段的双语索引、`BAAI/bge-m3` 模型版本 `5617a9f61b028005a4858fdac845db406aefb181`、本地图索引、`config/retrieval.yaml`、Top-5 和 3000 的近似消息 token 预算；本次在 CPU 上运行。

[逐题检索轨迹](../data/eval/dev_retrieval_trace.json)只保存 Top-5 片段 ID、英文片段 ID、逐组命中布尔值和运行配置，不含教材正文。题集 SHA256 为 `7830a7b00582fe336dbab61b9b001e339a5ec37b155fc22c1fbfa0e5f7193499`，向量索引指纹为 `65151e53aca9941c923c73334ee00d56364c3b26128760071cd015423330df9b`，图边与审核状态指纹为 `7b637f1f987421d69f9d462f2e613f34ad467fce20e78d6c197ec5632571ed60`。读者可在下载、校验、入库、构建索引和图谱后运行以下本地命令复核；若重新抽取或审核图边，图指纹可能不同，结果需重新统计：

```bash
python -m kgrag.cli eval --retrieval-only --questions data/eval/dev.jsonl --output data/runs/dev_retrieval.jsonl --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --device cpu
```

本次修订将自写查询/键/值片段列为 `dev-01` 的有效替代证据，并将明确说明束宽为 1 即贪心搜索的中文片段列入 `dev-09` 对应证据组。`dev-11` 和 `dev-12` 改为询问相对位置编码改动后与束宽为 7 时的确切 BLEU 结果；固定的中英文语料有相关方法说明，却没有这些实验数值。`typed_path` 虽让更多英文片段进入 Top-5，但在 `dev-02`、`dev-03`、`dev-10` 等题挤掉了所需证据，因此不能以英文片段数量推断检索质量提升。题目与候选证据组仍可在开发期复核，参数也只能在开发题上调整；正式比较须等封存测试集建立后进行。

同一索引和题集上，使用 `deepseek-flash`、`temperature=0`，完成 36 次答案生成调用。下表的“给出答案”仅指程序返回了非拒答文本；不能据此推断内容正确。

| 模式 | 可回答题给出答案 / 10 | 不可回答题经核查拒答 / 2 | 引用英文片段的答案 / 12 | 平均逐题延迟 | API 总 token |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dense` | 9 | 2 | 0 | 1.92 秒 | 29,286 |
| `one_hop` | 8 | 2 | 2 | 1.62 秒 | 28,539 |
| `typed_path` | 5 | 2 | 1 | 1.94 秒 | 28,716 |

[逐题答案运行摘要](../data/eval/dev_answer_summary.json)保留模式、拒答状态与形式、引用 ID、引用语言、延迟、模型版本、token 用量和与检索轨迹相同的题集/索引/图指纹；不保存教材或生成答案正文。完整逐题 JSONL 写入被忽略的 `data/runs/dev_eval_v2.jsonl`。两道不可回答题的六条输出均已逐条核查：三条使用标准拒答短句，三条明确说明缺少所问数值并附引用，均按拒答计。全部返回的引用 ID 都属于各题提供的证据；实际引用到的英文片段均带固定版本 URL 和行号。

`dev-08` 在三种模式都检出了两个必需证据组，但程序均返回拒答；`dev-09` 的 `one_hop` 与 `typed_path` 也在证据组全命中时拒答。这表明“检索命中”和“最终回答”必须分开检查。当前运行记录没有保留模型原始 JSON，无法区分模型主动拒答和生成后引用校验导致的拒答；下一步可在不记录教材正文或密钥的前提下增加拒答原因诊断。正式答案正确率、引用支持率和拒答准确率仍须按 Task 6 的封存题与盲评流程评定。

## Task 6：候选测试集与正式评测流程

`data/eval/test.jsonl` 现有 36 道候选题：单概念 8、多跳 16、比较 6、不可回答 6；加上开发集 12 道，达到设计稿规定的 48 道题型分布。候选题由助手依据本地固定版本中英文片段起草，证据 ID 均在当前 555 片段索引中存在，但每题仍标记 `pending_author_review`。这不是已经封存或经项目作者人工核查的测试集；不能据此发表正式测试指标。作者须逐题核对问题、金标准、每一组中英文可替代证据与不可回答边界，并修改或删除不合格题；确认至少一半题目由作者依据原文自行撰写，且题目没有由抽取图边反向生成。完成核查后再冻结题集文件和运行参数，记录 SHA256。

指标实现按必需论点组计算 `Recall@5`，一组中的任一经核查片段命中只计一次；多跳完整证据命中要求所有组均被命中。不可回答题从证据召回分母排除，在盲评表中单独核查拒答。汇总脚本给出三种模式的总体、题型、候选金标准证据语言和**实际检出证据语言**分组，以及未命中的证据组序号、平均逐题延迟和 API token 总量。金标准语言由候选片段元数据汇总，仍待作者核查；按实际检出语言分组只作描述，该分组会随检索模式变化，不能当成固定题目子集比较。

`scripts/score_annotations.py template` 将正式答案结果随机排序并去掉模式名，生成每题答案、引用和原文证据的评阅表；模式映射单独保存。评阅人逐题填 `answer_rating`（0 错误、1 部分正确、2 正确）、`citation_supported`、不可回答题的 `refusal_correct`，以及可选的错误类别和备注。`score` 只接受完整评分，并输出正确率、部分正确率、有引用答案的支持率、拒答准确率和失败案例。表中可能含第三方教材正文，**只能写入被 Git 忽略的 `data/runs/`**，不能提交。

封存并锁定模型、语料、图审核状态、提示词、Top-5 和 3000 近似 token 预算之后，仅对主测试集运行一次 `dense`、`one_hop`、`typed_path` 三路答案对照；随后用相同配置分别运行 `typed_path` 的 `β=0` 与关闭关系类型权重两种消融。命令如下，运行结果都留在本地：

```bash
python -m kgrag.cli eval --questions data/eval/test.jsonl --output data/runs/test_main.jsonl --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --device cpu
python -m kgrag.cli eval --questions data/eval/test.jsonl --modes typed_path --ablate-path --output data/runs/test_no_path.jsonl --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --device cpu
python -m kgrag.cli eval --questions data/eval/test.jsonl --modes typed_path --ablate-types --output data/runs/test_no_types.jsonl --embedding-revision 5617a9f61b028005a4858fdac845db406aefb181 --device cpu
python scripts/score_annotations.py retrieval --questions data/eval/test.jsonl --results data/runs/test_main.jsonl --output data/runs/test_retrieval_summary.json
python scripts/score_annotations.py template --questions data/eval/test.jsonl --results data/runs/test_main.jsonl --output data/runs/test_blind.jsonl --key data/runs/test_blind_key.jsonl
python scripts/score_annotations.py score --annotations data/runs/test_blind.jsonl --key data/runs/test_blind_key.jsonl --output data/runs/test_human_summary.json
```

目前没有封存集的三路主测试、消融或 108 条答案人工盲评分数，因此没有正式对照表。图边已有 40 条分层抽查，29 条通过、10 条拒绝、1 条待定，详见[图谱抽取抽查](./图谱抽取抽查.md)；由于分层且覆盖全部 7 条前置关系，29/40 只能作为抽查通过比例，不能当作 76 条活跃边的无偏 precision 估计。后续报告须保留 36 题规模、单一标注者、模型与语料版本以及该抽样限制。
