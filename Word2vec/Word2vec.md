---
title:Word2vec
categories: ML
tags: [Word2vec]
date: 2018-12-26
---

### 介绍

Word2vec通过学习文本然后用词向量的方式表征词的语义信息,然后使得语义相似的单词在嵌入式空间中的距离很近。而在Word2vec模型中有Skip-Gram和CBOW两种模式。

- Skip-Gram

  给定输入单词来预测上下文。

- CBOW

  是给定上下文来预测输入单词。

#### Word2vec 的两种优化方法

- Hierarchical Softmax

- Negative Sampling

  