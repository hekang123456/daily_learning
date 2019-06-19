---
title: Word2vec 优化方法之 Negative Sampling
categories: ML
tags: [Negative Sampling， Word2vec]
date: 2018-12-26
---

# Negative Sampling

### 1. 为什么要采用 Negative Sampling?
- 符号说明：
    |符号|含义|
    |----|----|
    |$n$|词表的大小|
    |$m$|词向量的长度|
    |$x_n$| $n$ 维的 one-hot 表示的词向量|
    
- 个人的理解：
  原始的 Word2vec 模型可以写成 $\text{Softmax}(x_n \times W_{n \times m} \times W_{m \times n})$。从输入层到输出层的变换 $x_n \times W_{n \times m}$ 可以仅仅看做是一个查表的过程，因为 $x_n$ 是 one-hot 的表示所以并不费时间。 <font color=red>在原始的 Word2vec 中从隐藏层到输出层的变化 $\hat{x_m} \times W_{m\times n}$ 中在每一次的训练中需要对 $W_{m \times n}$ 中的每一个参数做运算(无论是在前向计算或者反馈的过程中)， 即使采用了 Hierarchical Softmax 的方法，如果遇到一个生僻的词（这个词在 huffman 树的非常深的树叶节点处）作为中心词， 那么在每次训练过程中，需要进行的二元逻辑回归次数也是非常多的。  </font> 因此有必要找一种方法，这种方法能够使得每次训练的过程中只需要对一部分的参数进行训练，从而减少计算量，加快训练过程。Negative Sampling 就是一种这样的方法。

### 2. Negative Sampling 介绍

​	负采样的方法通过随机的采取中心词的若干个负样本，然后通过对正样本和负样本的<font color=red>二元逻辑回归</font>结果取极大似然作为目标对这些二元逻辑回归中的参数进行训练。也就是说，对于同一组的训练数据。我们只对这些负样本词和真的中心词为中心词的二元逻辑回归中的参数进行了训练。

​	**更通俗的将，我们对每个中心词都构造了一个二元逻辑回归，这个二元逻辑回归用于判断当前输入的上下文单词是否为该中心词的上下文。 那么如果词表的大小为$V$ 那么我们需要构造 $V$ 个二元逻辑回归。 如果我们有一个中心词 $w_0$ 以及其对应的上下文单词 $\text{context}(w)$，同时我们选取了 $n$ 个负样本为$w_1, w_2, ...., w_n$。 那么对应的目标函数就是** ：
$$
L = \sum\limits_{i=0}^n y_i \log (\sigma (x_{w_0}^T \theta^{w_i} ) ) + (1-y_i)\log (1- \sigma(x_{w_0}^T \theta^{w_i}))
$$
其中的 $x_{w_0} ^T$ 表示 $w_0$ 的上下文的输入。例如在 CBOW 就是上下文词向量取算术平均之后的结果， 在 Skip-gram 中就是上下文的单个词向量。当 $i=0$ 的时候 $y_i=1$ 否则为 0。



### 3. 负采样的方法

- 先将单位1均匀的划分为 $M$ 份。$M>>V$, $V$ 表示词表的大小。

- 计算每个单词占据整个词表的长度：
  $$
  len(w) = \frac{\text{count}(w)^{3/4}}{\sum\limits_{u \in \text{vocab} } \text{count}(u)^{3/4}}
  $$

- 根据计算得到的长度，将单位1中的小份分给每个单词。
- 随机在单位1中生成 $n$ 个点，这些点落在哪个单词所属的区域那就选这些单词作为负样本。

### 4. CBOW 和 Skip-gram 模型的算法步骤

##### 4.1 CBOW 模型

**输入：**基于 CBOW 的语料训练样本，词向量的维度大小 $Mcount$，CBOW的上下文大小是$2c$,步长$\eta$, 负采样的个数$neg$。

**输出：**词汇表每个词对应的模型参数$\theta$，所有的词向量$x_w$。

-    1.随机初始化所有的模型参数$\theta$，所有的词向量$w$；

- 2. 对于每个训练样本$(context(w_0),w_0)$,负采样出$neg$个负例中心词$w_i,i=1,2,...neg$；

- 3. 进行梯度上升迭代过程，对于训练集中的每一个样本$(context(w_0),w_0,w_1,...w_{neg})$做如下处理:

  - $e=0$, 计算 $x_{w_0} = \frac{1}{2c} \sum\limits_{i=1}^{2c} x_i$。

  - for  i to neg, 计算:
    $$
    f = \sigma(x_{w_0}^T \theta^{w_i}) \\
    g = (y_i -f)\eta \\
    e = e + g^{\theta^{w_i}} \\
    \theta^{w_i} = \theta^{w_i} + g x_{w_0}
    $$

  -  对于$context(w)$中的每一个词向量$x_k$(共$2c$个)进行更新：
    $$
    x_k = x_k + e
    $$

  - 如果梯度收敛，则结束梯度迭代，否则回到步骤3继续迭代。

##### 4.2 Skip-Gram模型

**输入：**基于 CBOW 的语料训练样本，词向量的维度大小 $Mcount$，CBOW的上下文大小是$2c$,步长$\eta$, 负采样的个数$neg$。

**输出：**词汇表每个词对应的模型参数$\theta$，所有的词向量$x_w$。

- 随机初始化所有的模型参数$θ$，所有的词向量$w$

-  对于每个训练样本$(context(w_0),w_0)$,负采样出$neg$个负例中心词$w_i,i=1,2,...neg$

-  进行梯度上升迭代过程，对于训练集中的每一个样本$(context(w_0),w_0,w_1,...w_{neg})$做如下处理：

  - a) for i = 1 to 2c:

    - $e$ = 0;

    - for j =0 to neg. 计算：
      $$
      f = \sigma(x_{w_0}^T \theta^{w_j}) \\
      g = (y_j -f)\eta \\
      e = e + g^{\theta^{w_j}} \\
      \theta^{w_j} = \theta^{w_j} + g x_{w_0}
      $$

    -  词向量更新
      $$
      x_ {w_{0i}} = x_{w_{0i}} + e
      $$

  - b) 如果梯度收敛，则结束梯度迭代，算法结束，否则回到步骤a继续迭代。

