---
title: Batch Normalization
categories: Deep Learning
tags: [Batch Normalization]
date: 2019-03-14
---

# 1. Batch Normalization
## 1.1 Batch Normalization 的做法
**输入：** mini-batch 输入 $x: B={x_1, ...., x_m}$ 
**输出：** 规范化后的网络响应 ${y_i = BN_{\gamma, \beta}(x_i)}$
- 计算 批数据的均值和方差
$$
\begin{align}
u_B &= \frac{1}{m} \sum_{i=1}^m x_i \\
\sigma_{B}^2 &= \frac{1}{m} \sum_{i=1}^m (x_i -u_{B})^2
\end{align}
$$
- 规范化
$$\hat{x_i} = \frac{x_i - u_B}{\sqrt{\sigma_B^2 + \epsilon } }$$
- 尺度变换和偏移
$$y_i = \gamma \hat{x_i} + \beta$$

## 1.2 Batch Normalization 的作用
- 是一种对抗梯度消失的有效手段 （将偏移0附加较大的数据分布拉回0均值1方差的分布中）
- 数据的分布一直处于敏感的区域，相当于不用考虑数据分布的变化。
- 因为数据本身的分布本身不是0均值1方差的，通过最后一步的偏移能够保证模型的非线性表达能力。

## 1.3 预测时的均值和方差怎么求
问题来源： 因为在预测的时候通常是单个样本进行预测的，因此没有所谓的批。
- 计算方法
$$
\begin{align}
E[x] &= E_{B}[u_B] \\
Var[x] &= \frac{m}{m-1} E_{B} [\sigma_B^2] \\
y &= \frac{\gamma}{\sqrt{Var[x] + \epsilon}} \cdot x + (\beta - \frac{\gamma E[x]}{\sqrt{Var[x] + \epsilon}}) 
\end{align}
$$
其中 均值和方差都是通过在训练集上得到的均值和方差进行求期望得到的。