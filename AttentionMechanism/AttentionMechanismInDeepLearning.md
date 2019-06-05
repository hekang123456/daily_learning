---
title: 深度学习中的注意力机制
categories: Deep Learning
tags: [Attention Mechanism, 注意力机制]
date: 2019-03-14
---

# 1. 深度学习中的注意力机制

> [来源](https://blog.csdn.net/tg229dvt5i93mxaq5a6u/article/details/78422216)

## 1.1 Encoder-Decoder 框架
<img src=AttentionMechanismInDeepLearning/encoder_decoder.png width=500  />

<center>图1.  抽象的文本处理领域的 Encoder-Decoder 框架</center>
<img src=AttentionMechanismInDeepLearning/rnn_encoder_decoder.png width=500 />

<center> 图2. RNN作为具体模型的Encoder-Decoder框架</center>


$$
\begin{align} 
\text{Source} &= <x_1, x_2, ..., x_m> \\  
\text{Target} &= <y_1, y_2, ..., y_n> \\
C  &= F(x_1, x_2, ..., x_m) \\
y_i &= \mathcal{G} (\mathbf{C}, y_1, y_2, ..., y_{i-1})  
\end{align}
$$

## 1.2 Attention 模型
<img src=AttentionMechanismInDeepLearning/encoder_decoder_attention.png width=500/>
$$
\begin{align}
y_i &= \mathcal{G} (\mathbf{C_i} , y_1, y_2, ..., y_{i-1}) \\
C_i &= \sum_{j=1}^{L_x} a_{ij} h_j
\end{align}
$$

- <font color=red> 注： Encoder-Decoder 框架中用于生成的隐藏内容$C$和Attention 中的隐藏内容$C_i$是不同的。 </font>

## 1.3 Attention 机制的本质思想
<img src=AttentionMechanismInDeepLearning/query_key_value.png width=500  />
<center> Attention 机制的本质思想 </center>
- **第一种解释**
可以将 Source 中的构成元素想象成一系列的 <Key, Value> 数据对构成。 而 Query 可以理解成来自 Target 中的某个元素， 通过计算 Query 和 Key 的相似度或者相关度，得到每个 Key 对应 Value的权重系数， 然后对 Value 进行加权求和。即得到最终的 Attention 数值。 所以对应的公式可以写成：
$$
Attention(Query, Source) = \sum_{i=1}^{L_x} Similarity(Query, Key_i) * Value_i
$$

- **第二种解释：** 软寻址
将Source 看做是存储器内存储的内容， 元素由 Key 和 Value 值组成。 当前有一个查询 Query。 通过 Query 和存储器内元素 key 进行相似性比较来寻址。 软寻址，指的是从每个 Key 中都会取出对应的 Value 值， 根据重要性，对 Value 进行加权求和。

## 1.4 注意力机制的三段式计算过程

<img src=AttentionMechanismInDeepLearning/attention_cal_proc.png width=500 />
<center>注意力机制的三阶段计算过程</center>
- 第一阶段： 根据 Query 和某个 $Key_i$， 计算两者的相似性或者相关性。 
  - 点积： $Similarity(Query, Key_i)= Query \cdot Key_i$
  - Cosine 相似性： $Similarity(Query, Key_i) = \frac{Query \cdot Key_i}{||Query|| \cdot ||Key_i||}$
  - MLP 网络： $Similarity(Query, Key_i)=MLP(Query, Key_i)$
- 第二阶段：对第一个阶段产生的分值，利用类似 SoftMax 的计算方法进行归一化， 同时能够突出重要元素的权重。
$$
a_i = Softmax(Sim_i) = \frac{e^{Sim_i}}{\sum_{j=1}^{L_x} e^{Sim_j}}
$$
- 第三阶段： 利用权值系数，对 Value 进行加权求和。
$$
Attention(Query, Source) = \sum_{i=1}^{L_x} a_i \cdot Value_i
$$

## 1.5 Self Attention 模型
<img src=AttentionMechanismInDeepLearning/self_attention.png width=500 />
<center>Self-Attention 的可视化</center>
- 理解
	- Self-Attention可以理解成 Source 和 Target 内容是一样的情况下的注意力机制。
	- Self-Attention可以捕获同一个句子中单词之间的一些句法特征或者语义特征。
	- Self-Attention能够更加容易捕获句子中长距离的相互依赖的特征。

