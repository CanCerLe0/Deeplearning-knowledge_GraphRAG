# Task 6 备选测试题（待作者挑选）

这 24 道题是 2026-11-25 根据固定版本中英文教材起草的**备选池**，未加入 `data/eval/test.jsonl`，也未运行测试集。参考答案和证据组只是审题辅助；可回答题仍需逐题核对论点与片段，不可回答题仍需在完整双语语料中确认边界。若选入正式题集，请按题型配额替换现有题，不要简单追加。助手起草的题目即使被选中，也不能计入设计稿要求的“至少一半由作者自行撰写”。

每道题的“依据”按必需论点组列出；同组多个片段是待人工确认的替代证据。题号用于回复选择结果。

## 单概念

| 题号 | 候选问题 | 参考答案 | 原文定位 |
|---|---|---|---|
| B01 | 视觉注意力类比中，红色杯子为什么会产生非自主性的注意力提示？ | 它因颜色和周围黑白物品形成鲜明对比，突出性引发不由自主的注意。 | 第1组：[81580be4 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md#L29-L58) |
| B02 | Bahdanau 注意力在一个解码时间步把什么作为查询、键和值？ | 上一时间步的解码器隐状态作为查询；各时间步的编码器隐状态同时作为键和值。 | 第1组：[a1db6e3a · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/bahdanau-attention.md#L443-L444) |
| B03 | 固定正弦位置编码的偶数维和奇数维分别使用什么函数？ | 偶数维使用正弦，奇数维使用余弦。 | 第1组：[4b4474b6 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/self-attention-and-positional-encoding.md#L145-L163) |
| B04 | 齐普夫定律如何描述词频与频率排名的关系？ | 第 i 高频词的频率近似与排名 i 的某个幂次倒数成正比。 | 第1组：[4da4f03d · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/language-models-and-dataset.md#L183-L229) |
| B05 | 若语言模型始终在整个词表上均匀预测，其困惑度是多少？ | 等于词表中不同词元的数量。 | 第1组：[f5155096 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/rnn.md#L277-L300)、[41ac1256 · en](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recurrent-neural-networks/language-model.md#L215-L215) |
| B06 | 高斯核注意力的宽度缩小时，回归估计会怎样变化？ | 估计通常不那么平滑，但更能跟随局部变化。 | 第1组：[30c77e89 · en](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_attention-mechanisms-and-transformers/attention-pooling.md#L205-L225) |

## 多跳

| 题号 | 候选问题 | 参考答案 | 原文定位 |
|---|---|---|---|
| B07 | 从视觉注意力的两种提示出发，注意力机制中的查询和键各对应什么？ | 自主性提示对应查询；与值配对、反映感官输入突出性的非自主性提示对应键。 | 第1组：[81580be4 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md#L29-L58)<br>第2组：[4299ae20 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md#L64-L92) |
| B08 | Bahdanau 注意力怎样选择源序列的信息？这种改动对教材中的训练速度有何影响？ | 它按解码状态对编码器各位置加权，选择相关输入；教材指出加入注意力后训练慢于无注意力的 seq2seq。 | 第1组：[a1db6e3a · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/bahdanau-attention.md#L443-L444)<br>第2组：[20d05770 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/bahdanau-attention.md#L363-L395) |
| B09 | 多头注意力如何合并各头输出，为什么实现时会把每头维度设为总输出维度除以头数？ | 各头结果连接后经线性变换；缩小每头维度避免参数与计算代价随头数大幅增长，并支持并行。 | 第1组：[ce1dccad · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/multihead-attention.md#L30-L74)<br>第2组：[a8ed54bd · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/multihead-attention.md#L88-L94) |
| B10 | 固定正弦位置编码如何表示绝对位置，又为何能表示固定偏移的相对位置？ | 每个位置按不同频率的正弦、余弦形成向量；固定偏移后的位置编码可由原位置编码线性变换得到。 | 第1组：[4b4474b6 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/self-attention-and-positional-encoding.md#L145-L163)<br>第2组：[2b1d767b · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/self-attention-and-positional-encoding.md#L340-L359) |
| B11 | 教材的英法机器翻译数据为何是成对文本，源语言与目标语言词表怎样建立？ | 每条数据是一对英语源文本和法语目标文本；两种语言分别建立词表。 | 第1组：[5a7673c8 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/machine-translation-and-dataset.md#L65-L94)<br>第2组：[a1db3523 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/machine-translation-and-dataset.md#L169-L184) |
| B12 | 完整的序列联合概率怎样按链式法则分解？二元语法又删去了哪些条件？ | 链式法则让每个词元依赖所有前文；二元语法近似只依赖紧前一个词元。 | 第1组：[9c98f9e8 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/language-models-and-dataset.md#L34-L72)<br>第2组：[f882f533 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/language-models-and-dataset.md#L104-L122) |
| B13 | 词频长尾和稀有 n 元组为什么让简单计数与拉普拉斯平滑难以胜任语言建模？ | 齐普夫长尾使低频词估计困难；许多 n 元组出现极少，简单平滑会高估尾部或难以处理稀疏组合。 | 第1组：[4da4f03d · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/language-models-and-dataset.md#L183-L229)<br>第2组：[94882557 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/language-models-and-dataset.md#L231-L239)、[8f50dba4 · en](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_recurrent-neural-networks/language-model.md#L151-L167) |
| B14 | 扩大 n 元语法的历史窗口有什么存储代价？循环神经网络怎样避免参数随时间步增加？ | n 元语法需要存储的组合数随 n 指数增长；循环神经网络用隐状态和跨时间步共享的参数。 | 第1组：[ee23d85a · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/rnn.md#L4-L39)<br>第2组：[88ae1aef · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/rnn.md#L115-L148) |
| B15 | 基本循环 seq2seq 若用编码器最后一个隐状态作为上下文，解码器每一步还依赖什么？ | 最后一个编码器隐状态概括输入；解码器还依赖先前输出词元及上一解码器隐状态。 | 第1组：[c9371347 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/seq2seq.md#L85-L127)<br>第2组：[60b005ea · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/seq2seq.md#L296-L328) |
| B16 | 序列概率可倒序分解，为什么时间因果任务仍偏向正向建模？训练集切分又应遵守什么？ | 数学上可倒序分解，但未来不能改变过去，正向因果解释更自然；训练应尊重时间顺序，不使用未来观测。 | 第1组：[ab18b8b6 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/sequence.md#L140-L157)<br>第2组：[99ad1c62 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-neural-networks/sequence.md#L528-L531) |

## 比较

| 题号 | 候选问题 | 参考答案 | 原文定位 |
|---|---|---|---|
| B17 | 人类视觉注意力中的非自主性提示与自主性提示，触发选择的依据有何不同？ | 前者由物体突出性引发，后者受目标、认知和意愿控制。 | 第1组：[81580be4 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md#L29-L58)<br>第2组：[4299ae20 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/attention-cues.md#L64-L92) |
| B18 | 非参数 Nadaraya-Watson 汇聚与带可学习参数的版本，查询和键的距离权重有什么不同？ | 非参数版本直接用核函数给距离加权；带参数版本在距离上引入可学习缩放参数。 | 第1组：[51e406cc · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/nadaraya-waston.md#L177-L213)<br>第2组：[806c3022 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_attention-mechanisms/nadaraya-waston.md#L315-L326) |
| B19 | 常数核与高斯、Boxcar、Epanechikov 核在教材的回归例子中有什么差别？ | 常数核退化为输出均值，估计不理想；另外三种非平凡核按位置加权，得到较可用的估计。 | 第1组：[3f9ac03e · en](https://github.com/d2l-ai/d2l-en/blob/23d7a5aecceee57d1292c56e90cce307f183bb0a/chapter_attention-mechanisms-and-transformers/attention-pooling.md#L194-L201) |
| B20 | 教材的字符级语言模型与英法机器翻译在词元化粒度和词表规模上有什么取舍？ | 字符级粒度较细、词表较小；机器翻译例子更倾向单词级，词表更大并需处理低频词。 | 第1组：[82ea23cb · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/machine-translation-and-dataset.md#L118-L147)<br>第2组：[a1db3523 · zh](https://github.com/d2l-ai/d2l-zh/blob/e6b18ccea71451a55fcd861d7b96fddf2587b09a/chapter_recurrent-modern/machine-translation-and-dataset.md#L169-L184) |

## 不可回答

| 题号 | 候选问题 | 参考答案 | 原文定位 |
|---|---|---|---|
| B21 | 把英文教材核回归例子的高斯核宽度改为 0.3 后，验证集均方误差的确切值是多少？ | 证据不足，无法根据固定语料回答这一未运行配置的数值。 | 无必需证据组；需核查双语语料边界 |
| B22 | 把教材 Transformer 的注意力头数由 4 改为 8 后，英法翻译 BLEU 精确变化多少？ | 证据不足，无法根据固定语料回答这一改动的实验结果。 | 无必需证据组；需核查双语语料边界 |
| B23 | 在完整测试集上，Bahdanau 注意力相对无注意力 seq2seq 的 BLEU 提高了多少？ | 证据不足，固定语料只有局部实验说明，没有这一完整测试集差值。 | 无必需证据组；需核查双语语料边界 |
| B24 | RoPE 旋转位置编码的变换矩阵具体是什么？ | 证据不足，当前固定语料没有给出 RoPE 的变换矩阵。 | 无必需证据组；需核查双语语料边界 |
