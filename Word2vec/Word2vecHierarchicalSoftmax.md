---
title: Word2vec 优化方法之 Hierarchical Softmax
categories: ML
tags: [Hierarchical Softmax， Word2vec, 层次Softmax]
date: 2018-12-26
---

# Hierarchical Softmax

> [来源](https://www.cnblogs.com/pinard/p/7243513.html)
### 1. 为什么要采用层次Softmax?
- 普通的 word2vec 可以写成 $\text{Softmax}( x_{n} \times  W_{n \times m} \times W_{m \times n} )$ 。 
- 从输入层到隐藏层的变换 $x_n  \times W_{n \times m}$ 因为输入是 one-hot 的表示方法， 因此这个变换只需要选出 1 所在的那一行参数，因此时间复杂度 $O(m)$ 的。
- 从隐藏层到输出层的变换 $\hat{x_m} \times  W_{m \times n}$, 其中 $\hat{x_m}$ 表示隐藏层的神经元（向量）。 需要考虑 $m \times n$ 个参数。因此 hierarchical softmax 用于<font color=red>减少隐藏层到输出层的计算复杂度。</font>

### 2.层次Softmax 的做法？
- 对词表中的所有单词按照其在文本库中的出现的频率构造一颗 **赫夫曼树**。
  - 赫夫曼数的叶结点都表示一个单词。
  - 每个根结点表示概率值。根结点的概率值为1。
- 在赫夫曼树的每个结点都带有参数 $\theta_i$, $\theta_i \in R^{m \times 2}$ 
- $\theta_i$ 用来计算隐藏层在**该结点的时候的输出概率**, 采用的是 Sigmoid 函数。 $\sigma (\hat{x_m} \theta_i )$ 当这个概率值大于 0.5 的情况下，可以人为的定义其属于左结点或者右结点。
- 因此 采用 hierarchical softmax 在隐藏层到输出层的参数总量大约为 $ 2m \log_2(V) $ 这个数量比原本的 参数量 $m \times n$ 少的多。

### 3.基于 hierarchical softmax 的模型梯度计算。
- 符号说明
    | 符号  | 含义 |
    | ---- | ---- |
    |$w$|输入的词向量|
    |$x_w$ |隐藏层的向量（Huffman tree 根结点的词向量）|
    |$l_w$| 从根结点开始到 $w$ 的叶子结点，包含的结点总数是 $l_w$ |
    |$d_j^w$| $d_j^w \in \{0, 1\}$, 表示第 $j$ 个结点的赫夫曼编码， $j=2,3,...,l_w$ 因为根结点没有对应的编码，所以 $j$ 是从2开始的。|
    |$\theta_{j}^w$| 表示第 $j$ 个结点的参数，该参数与 $\sigma(x_w \theta_{j}^w)$ 用来表示下一个结点是左结点还是右结点的概率。$j=1,2,...,l_w-1$ 因为叶结点是不含参数的所以到 $l_w-1$| 为止。|

- 过程

  - 在 huffman tree 左结点或右**结点的概率**是：
    $$
    P(d_j^w|x_w, \theta_{j-1}^w ) = 
    \left\{ \begin{align}
    &\sigma(x_w^T \theta_{j-1}^w) &d_j^w = 0\\
    &1-\sigma(x_w^T \theta_{j-1}^w) &d_j^w = 1
    \end{align} \right.
    $$
  - **$w$对应的赫夫曼编码的似然函数**为：
  $$
  \prod_{j=2}^{l_w} \sigma(x_w^T \theta_{j-1}^w)^{1-d_j^w} (1-\sigma(x_w^T \theta_{j-1}^w )^{d_j^w})
  $$
  - 对数似然
    $$
    \begin{align}
    L &= \log \prod_{j=2}^{l_w} P(d_j^w | x_w, \theta_{j-1}^w) \\
      &= \sum\limits_{j=2}^{l_w} [ (1-d_j^w)\log( \sigma(x_w^T \theta_{j-1}^w) ) + d_j^w \log( 1- \sigma(x_w^T \theta_{j-1}^w))]
    \end{align}
    $$
  - 求导
  $$
  \begin{align}
  \frac{\partial L }{\partial \theta_{j-1}^w} &=(1-d_j^w) \frac{\sigma(x_w^T \theta_{j-1}^w) (1- \sigma(x_w^T \theta_{j-1}^w))}{\sigma(x_w^T \theta_{j-1}^w)} x_w^T - d_j^w \frac{ \sigma(x_w^T \theta_{j-1}^w) (1- \sigma(x_w^T \theta_{j-1}^w))}{1- \sigma(x_w^T \theta_{j-1}^w)}x_w^T \\
  & = (1-d_j^w - \sigma(x_w^T \theta_{j-1}^w))x_w \\
  \text{同理可得：} \\
  \frac{\partial L }{\partial x_w} &= \sum\limits_{j=2}^{l_w} (1-d_j^w - \sigma(x_w^T \theta_{j-1}^w)) \theta_{j-1}^w
  \end{align}
  $$

### 4. 基于 Hierarchical Softmax 的 CBOW 模型
- **符号说明**

    |符号| 含义|
    |-----|----|
    |$c$| CBOW模型中上下文的大小是 $2c$|
    |$x_i$|$x_w$ 的第 $i$ 个上下文单词|
    |$eta$| 步长、学习率|

- **对每个训练样本的训练过程**

  - **求和：** $ \frac{1}{2c} \sum\limits_{i=1}^{2c} x_i $。 CBOW 需要将上下文的词向量通过加和转化为一个词向量。
  - **e=0:** 用于对每个结点的梯度进行加和，因此对 $x_w$ 进行更新的时候用到了这个。 
  - 对 $j=2,3...,l_w$ 分别计算对应结点参数的梯度
    $$
    \begin{align}
    f &= \sigma(x_w^T \theta_{j-1}^w) \\
    g &=(1-d_j^w-f)\eta \\
    e &=e+g\theta_{j-1}^w \\
    \theta_{j-1}^w &= \theta_{j-1}^w + gx_w
    \end{align}
    $$
  - **对 $x_i$ 进行梯度更新**。
  	$$ x_i = x_i + e$$
  	<font color=red>hk:</font> 注意这里的 $e$ 严格来说应该是对 $x_w$ 求得的梯度，但是 每个 $x_i$ 和这个梯度相比只差了个常数的倍数 $\frac{1}{2c}$。 因此这里直接用 $e$ 来更新 $x_i$。
  - 直到梯度收敛的时候结束，否则继续迭代

### 5. 基于 Hierarchical Softmax 的 skip-gram 模型
- **训练过程中采用上下文来预测中心词**
  <font color=red> skip-gram 的核心思想是通过 中心词对上下文进行预测也就是 $P(x_i|x_w), i=1,2,...,2c$。因为上下文是相互的，因此我们在期望 $P(x_i|x_w)$ 最大的同时也是要求 $P(x_w | x_i)， i=1,2,...,2c$ 最大。 因为对 $P(x_w |x_i)$ 求期望最大的过程我们同时对2c个单词进行更新，能够使整体的迭代更加均衡。所以，skip-gram 相当于对 2c个输出进行迭代更新。</font>

- **对于每一个训练样本 $(x_w, x_i)$进行迭代**
  - **e=0**
  - 对于 $j$从1到 $l_w$:
  $$
  \begin{align}
  f &= \sigma(x_i^T \theta_{j-1}^w) \\
  g &= (1-d_j^w -f)\eta \\
  e &= e + g\theta_{j-1}^w \\
  \theta_{j-1}^w &= \theta_{j-1}^w + g_{x_i} 
  \end{align}
  $$
  - $x_i = x_i + e$

- 如果梯度收敛就，结束梯度迭代，算法结束，否则，继续。

### 6. 一些个人的理解
- 在仅仅考虑训练的过程，我们会发现在引用了hierarchical softmax 的word2vec 中 skip-gram 和 CBOW 的输出都是对中心词的预测， 只不过 CBOW一次性 的输出 2c 个词然后在隐藏层进行了一个 加和构成一个向量，然后来预测对应的输出，而skip-gram 一次只输入一个单词，来预测对应的输出，因此我们可以理解为 <font color=red> CBOW一次训练更新了 2c个词向量对应的权重值，而 skip-gram一次只更新对应一个单词的权重。 或者可以理解为 CBOW用上下文的2c个单词来预测中心词，而skip-gram 利用上下文的每一个单词来预测中心词。</font>

