---
title: 提升方法 
categories: ML
tags: [AdaBoost, GBDT, 提升树, 前向分布算法]
date: 2019-01-05
---

# 提升方法 

#### 介绍

- 强可学习与弱可学习
  - 强可学习 : 一个概念（类）,存在一个多项式的学习算法能够学习他，并且正确率很高，则这个概念是强可学习的。
  - 弱可学习：一个概念（类）,存在一个多项式的学习算法能够学习他，但是学习的正确率只比随机猜测略好，那么这个概念就是弱可学习的。
  - 强可学习和弱可学习是等价的， 一个概念是强可学习的充分必要条件是这个概念是弱可学习的。

- 提升方法的由来

  在学习中， 如果发现了 "弱学习算法" 那么能够将它提升为 "强学习算法"。

- 提升方法的两个问题
  - 在每一轮如何改变训练数据的权重值或概率分布？
  - 如何将弱分类器合成一个强分类器？
  
- 注： <font color=red>提升方法实际采用加法模型（即基函数的线性组合）与前向分布算法。</font> 所以对于上面的两个问题我们可以改为 （1）确定损失函数，如何求的基本分类器，也就是说输入数据变了训练得到的基本分类器也不一样了 （2）怎么确定基本分类器在合成整个分类器中基本分类器对应的权重。



#### 加法模型与前向分布算法

- 加法模型：  
  $$
  f(x) = \sum\limits_{m=1}^M \beta_m b(x; \gamma_m) 
  $$
  其中， $b(x; \gamma_m)$ 是基函数， $\gamma_m$ 是基函数的参数，$\beta_m$ 是基函数的参数。 也就是将一些基本分类器按照权值相加得到一个分类器。

- 前向分布算法

  - 作用

    学习加法模型, 可以理解为是给定训练数据和损失函数的条件下，学习加法模型 $f(x)$ 成为经验风险最小化的问题。
    $$
    \min_{\beta_m, \gamma_m} \sum\limits_{i=1}^N L \left( y_i,  \sum\limits_{i=1}^N \beta_m b(x_i; \gamma_m) \right)
    $$
    这是一个复杂的问题，因此我们可以用前向分布算法从前往后，每一步只学习一个基函数与系数，逐步逼近优化目标的函数式子。

  - 步骤

    **输入:** 数据集 $T=\{ (x_1, y_1), (x_2, y_2), ..., (x_N, y_N) \}$ ； 损失函数 $L(y, f(x))$; 基函数集 $\{ b(x; y) \}$

    **输出：**加法模型 $f(x)$

    - 初始化 $f_0(x) = 0;$

    - 对 $m =1,2,3..., M$

      - $(\beta_m, \gamma_m) = \arg \min\limits_{\beta, \gamma} \sum\limits_{i=1}^N L( y_i, f_{m-1}(x_i) + \beta b(x_i; \gamma))$
      - $f_m(x) = f_{m-1}(x) + \beta_m b(x; \gamma_m)$

    - 得到加法模型
      $$
      f(x) = f_M (x) = \sum\limits_{m=1}^M \beta_m b(x; \gamma_m)
      $$
      

#### AdaBoost

​	AdaBoost 通过在每次训练过程中根据当前分类器对数据分类是否正确，调整训练数据的权重值。 通过当前分类器在训练数据集上的分类精度判断当前分类器的权重，最后将所有分类器进行加权求和得到最终的分类器。

- AdaBoost 的计算流程

  <img src="/boosting/adaboost.png" /> 
  
  - 初始数据权值：  $w_{1i} = \frac{1}{N}, i=1,2,...,N$
  
  - 加权误差：  $e_m = \sum\limits_{i=1}^N P(G_m(x_i) \neq y_i) = \sum\limits_{i=1}^N w_{m_i} I( G_m(x_i) \neq y_i)$  其中 $G_m (x)$ 表示基本分类器，结果为-1或1。
  
  -   分类器权值系数： $\alpha_m = \frac{1}{2} \log \frac{1-e_m}{e_m}$
  
  - 数据权值分布： 
    $$
    \begin{align}
    w_{m+1, i} &= \frac{w_{mi}}{z_m} \exp(-\alpha_m y_i G_m (x_i)), i=1,2,...,N \\
    Z_m &= \sum\limits_{i}^N w_{mi}\exp(-\alpha_m y_i G_m(x_i))
    \end{align}
    $$
  
  - 当前分类器： $f_m (x) = \sum\limits_{i=1}^M  \alpha_m G_m (x)$
  - 停止条件： 当前分类器$f_m(x)$ 的误差是否小于某个指定的值。

- 分类器权值系数和数据权值分布的说明

  - **权值系数:** 分类误差越大，权值越小，  $\frac{1}{2}$ 是一个分界点，小于 0.5 时大于0， 大于0.5时小于0。
  - **数据权值分布：** 如果正确分类权值为  $\frac{w_{mi}}{ez_m}$ 错误分类权值为 $\frac{ew_{mi}}{z_m}$ 。说明错误分类的数据在下一个分类器中的重要程度在不断的提升，正确分类的数据的重要程度在下降。

- 优缺点

  - 优点：可以充分利用不同分类算法的优势进行建模，也可以将同一算法的不同设置进行组合。
  - 缺点： 只支持二分类， 多分类需要借助 one-versus-rest 思想。

- 说明

  AdaBoost 是前向分布算法的特列，模型由基本分类器组成的加法模型， 损失函数是指数损失函数。

  

#### 提升树

​	提升方法采用的式加法模型（即基函数的线性组合）与前向分布算法，以**决策树为<font color=red>基函数</font>**的提升方法。这里的决策树是只有两个节点叶节点的简单决策树，即所谓的<font color=red>**决策树桩**</font>。

- 类别
  - 分类：  采用二叉分类树作为基函数，损失函数采用的是指数损失函数，可以将分类决策树视为 AdaBoost 的基本分类器为决策树的一种特殊情况。
  - 回归： 采用二叉回归树作为基函数。

- 回归问题的提升树算法

  **输入：** 训练数据集 $T = \{ (x_1, y_1), (x_2, y_2), ..., (x_N, y_N)\}$

  **输出:**  提升树 

  - 初始化 $f_0(x) = 0$

  - 对 $m=1,2,...,M$

    - 计算残差
      $$
      \begin{align}
      T(x; \theta) &= \sum\limits_{j=1}^T c_j I(x \in R_j) \\
      r_{mi} &= y_i - f_{m-1}(x_i), i=1,2,...,N
      \end{align}
      $$

    - 拟合残差

      利用回归树拟合残差 $r_{mi}$, 得到 $T(x; \theta_m)$。

    - 更新 : $f_m(x) = f_{m-1}(x) + T(x; \theta_m)$

  - 得到回归问题的提升树

    $f_M(x) = \sum\limits_{m=1}^M T(x; \theta_m)$

#### 梯度提升方法

​	当采用前向分布算法求解加法模型的时候，如果损失函数是平方损失或者指数损失函数的时候，每一步的优化都很简单。分别对应于残差与AdaBoost 中对训练数据权重的调节。 但是对于一般的损失函数，每一步的优化比较困难**(<font color=purple>为什么比较困难，不能用残差吗？</font>)**。 因此提出了梯度提升算法。 

​	**这种方法关键在于利用损失函数的负梯度在当前模型中的值作为回归问题梯度提升算法中的残差的近似值，拟合一个回归树。**

- 梯度提升算法

  **输入:** 训练数据集 $T= \{ (x_1, y_1), (x_2, y_2),...,(x_N, y_N) \}$, 损失函数为 $L(y, f(x))$。

  **输出:** 回归树 $\hat{f(x)}$。

  -  初始化 
    $$
    f_0(x) = \arg \min\limits_c \sum\limits_{i=1}^N L(y_i ,c)
    $$

  - 对 $m =1,2,...,M$

    - 对 $i=1,2,...,N$ 计算
      $$
      r_{mi} = - \left[ \frac{\partial L(y_i, f(x_i))}{\partial f(x_i)} \right]_{f(x) = f_{m-1}(x)}
      $$

    - 对 $r_{mi}$ 拟合一个回归树，得到第 $m$ 棵树的叶节点区域 $R_{mj}, j=1,2,...,J$。

    - 对 $j=1,2,...,J$, 计算
      $$
      c_{mj} = \arg \min\limits_{c} \sum\limits_{x_i \in R_{mj} L(y_i, f_{m-1}(x_i)+c)}
      $$

    - 更新 $f_m(x) = f_{m-1}(x) + \sum\limits_{j=1}^J c_{mj} I(x \in R_{mj})$

  - 得到回归树
    $$
    \hat{f(x)} = f_M(x) = \sum\limits_{m=1}^M \sum\limits_{j=1}^J c_{mj} I(x \in R_{mj})
    $$

- 对于平方损失函数，负梯度就是残差，对于一般的损失函数，它就是残差的近似值。



#### 梯度提升树（GBDT, Gradient Boosting Decison Tree）

- 介绍

  - 提升树 + 梯度提升
  - 弱学习器限定只能使用 CART 树

  -  可以用来做分类也可以用来做回归

- GBDT 回归算法

  和提升方法中的步骤是一样的， 只是限定了弱学习器（基函数）只能采用 CART 树。

- GBDT 分类算法

  因为 GBDT  做分类的时候样本输出不是连续的值，因此可以<font color=red>拟合预测的概率值: hk</font>。 可以采用指数损失函数或者对数损失函数， 当采用指数损失函数的时候退化为 AdaBoost 算法。 对于对数损失函数又有二元分类和多分类的区别。

  - 二元 GBDT 分类算法

    - 损失函数为： $L(y, f(x)) = \log(1+exp(-yf(x)))$ 其中 $y \in  \{-1, +1\}$

    - 负梯度为：$r_{ti} = - \frac{\partial L(y_i, f(x_i))}{ \partial f(x_i)} = \frac{y_i}{1+ \exp(y_i f(x_i))}$ 
    - 叶节点的最佳负梯度拟合值是： $c_{tj} = \arg\min\limits_{c}  \sum\limits_{x_i \in R_{tj}} log(1+ \exp(-y_i (f_{t-1}(x_i)+c)))$
    - 最佳负梯度的近似值是：  $c_{tj} = \frac{  \sum\limits_{x_i \in R_{tj}}   r_{tj} }{  \sum\limits_{x_i \in R_{tj}  } |r_{ti}| (1-|r_{ti}| )  }$

  - 多元 GBDT 分类算法

    - 损失函数： $L(y, f(x)) = - \sum\limits_{k=1}^K y_k \log p_k(x)$ 
    - 预测为第 $k$ 类的表达式是： $p_k (x) = \frac{ \exp(f_k (x)) }{ \sum\limits_{l=1}^K \exp(f_l(x)) }$
    - 负梯度： $r_{til} =  -\left[ \frac{\partial  L(y_i, f(x_i))}{ \partial f(x_i) } \right]_{f_k (x) = f_{l, t-1} (x)} =y_{il} - p_{l,t-1}(x_i) $
    - 最佳负梯度拟合值是： $ c_{tjl} = \arg \min\limits_{c_{jl}} \sum\limits_{i=0}^m \sum\limits_{k=1}^K L(y_k, L(y_k, f_{t-1, l}(x) + \sum\limits_{j=0}^J c_{jl} I(x_i \in R_{tjl}) )) $ 
    - 最佳负梯度的近似值： $ c_{tjl} = \frac{K-1}{K} \frac{ \sum\limits_{x_i \in R_{tjl} }r_{til} }{ \sum\limits_{x_i \in R_{tjl} } |r_{til}| + (1- |r_{til}|) } $

  - GBDT 的正则化

    - 添加正则化项（或步长）
      $$
      f_k(x) = f_{k-1}(x) + vh_k (x)
      $$

    - 通过子采样（无放回），只用小部分样本去做决策树的拟合 （**<font color=red>具体怎么做？</font>**）

    - 对CART 回归树做正则化剪枝

  

  #### XGBoost

  

