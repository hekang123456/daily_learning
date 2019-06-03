---
title:GloVe
categories: ML
tags: [GloVe]
date: 2018-12-26
---

# GloVe 的计算方法

- 构造共线矩阵
  $X_{ij}$: 表示单词 $i$ 在窗口 $d$ 内出现单词 $j$ 的次数。GloVe 在此基础上提出了一个衰减函数 $1/d$。表示在窗口内出现的单词也会根据远近有不同的重要性。 
- 构造词向量和共线矩阵之间的近视关系。
  <font color=red>$w_i^T \hat{w_j} + b_i + \hat{b_j} = \log(X_{ij})$</font>
- 构造损失函数
    $$
    J = \sum\limits_{i,j=1}^V f(X_{ij}) (w_i^T \hat{w_j} +b_i +\hat{b_j} -\log(X_{ij}))^2 \\
    f(x) = \left\{   \begin{align} &(x/x_{max})^{\alpha}  & if \ x<x_{max}\\ &1 &otherwise  \end{align} \right.
    $$
    $\alpha$ 的取值一般是 0.75， 而 $x_{max}$ 的取值是100。 $f(x)$ 表示出现越多的单词权重越大，但是也不应该太大，因此超过一定的数以后就不增长了。
- 采用梯度下降方法求解 $w_i $ 和 $\hat{w_i}$
	从原理上 $w_i$ 和 $\hat{w_i}$ 应该是一样的，但是因为初始值的不同，导致结果有差别。
- 最终的结果
	$w+\hat{w}$

