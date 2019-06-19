---
title:日常记录的一些问题
categories: ML
tags: [pooling, 池化操作]
date: 2019-06-19
---

####  max pooling 和 average pooling 的区别？

- max-pooling 相比于 average pooling 提供了**非线性**。
- 分类问题上max-pooling相当于**特征选择**，选出有利于分类辨识度高的特征去分类。而 average pooling 相当于**特征的融合**，average pooling 能够减小估计值的**方差**，更多的保留图像的背景信息（在文本中理解为一通用的信息？）。max pooling 能够减小**偏差**。更多的保留图像的背景信息。 
- averaging pooling 的作用更主要体现在减少参数维度的贡献上更大一点，以及**信息的完整传递上**。

#### pooling 的作用？

- 减参
- 控制过拟合
- 降维
- 提供了 平移，旋转的不变性。
- 减少计算量
- 增大感受野

