---
title: 感知器学习算法
date: 2018-11-29
categories: [ML, 统计学习方法]
tags: [感知器, 线性分类， 对偶问题]
---

# 1 感知机
## 1.1 感知机模型
- 定义： 
$$
  \begin{align}
    f(x) &= \text{sign} (wx+b) \\
    \text{sign}(x) &= \begin{cases} +1, x \leq 0 \\ -1, x<0 \end{cases}
  \end{align} 
$$
- 属性： 
  - 线性分类模型， 判别模型
  - 假设空间： 所有线性分类模型

- 损失函数（经验风险损失）：
  $$L(w,b) = -\sum\limits_{x_i \in M} y_i (wx_i +b)$$	
  其中 $M$ 是误分类点的集合。 因为对于 $(wx_i +b) >0$ 的情况其预测标签为1， 对于误分类的情况实际标签为$-1$。 因此，需要使得 $(wx_i +b)​$ 往0处靠近。 所以可以采用这个作为损失函数。 

## 1.2 感知器学习算法

- 方法： 随机梯度下降算法(stochastic gradient descent)

- 梯度： 
  $$
  \begin{align}
  	\Delta_w L(w,b) &= -\sum_{x_i \in M} y_i x_i \\
  	\Delta_b L(w,b) &= -\sum_{x_i \in M} y_i
  \end{align}
  $$

- 梯度的更新
  随机的选择一个误分点进行更新：
  $$
  \begin{align}
  	w &= w+\eta y_i x_i \\
  	b &= b+\eta y_i 
  \end{align}
  $$
  $0<\eta\leq1$ 在统计学习中又称为学习率.

- 例子1

  如图2.2所示的训练数据集， 其正实例点是$x_1＝(3,3)^T， x_2＝(4,3)^T​$， 负实例点是$x_3＝(1,1)^T​$， 试用感知机学习算法的原始形式求感知机模型$f(x)＝\text{sign}(w·x+b)​$。 这里， $w＝(w^{(1)},w^{(2)})T， x＝(x^{(1)},x^{(2)})$

  - 代码：

  ```Python
  import numpy as np
  x = np.matrix([[3, 3], [4, 3], [1, 1]])
  y = [1, 1, -1]
  w = np.matrix([[0], [0]])
  b = 0
  lr = 1
  flag  = 0
  while flag<3:
      flag = 0
      for i in range(len(data)):
          tmp = (data[i]*w + b) * y[i]
          if (tmp[0, 0] <= 0): # 被误分
              w = w + lr*y[i]*x[i].T
              b = b + lr*y[i]
              print(w.T, b)
          else:
              flag += 1
  ```

  - 结果：

    ```
    [[3 3]]
    [[3 3]] 1
    [[2 2]] 0
    [[1 1]] -1
    [[0 0]] -2
    [[3 3]] -1
    [[2 2]] -2
    [[1 1]] -3
    ```

## 1.3 算法的收敛性
## 1.4 感知器学习算法的对偶形式
### 1.4.1 说明：
感知器采用随机梯度下降的方法进行梯度下降的迭代过程如下：
$$
\begin{align}
	w &= w + \eta y_i x_i \\
	b &= b + \eta y_i 
\end{align}
$$
在经过$n$ 次的迭代之后，$w$ 和 $b$ 受到第$i$个样本$(x_i ,y_i)$的影响所改变值分别是 $n_i \eta y_i x_i $ 和 $n_i \eta y_i$。$n_i$表示第 $i$个样本出现错判的次数。 $\eta$ 是学习率。 用$\alpha_i$ 表示$n_i \eta$ 因此，不难得到最后的$w$ 和 $b$ 分别是：
$$
\begin{align}
	w &= \sum\limits_{i=1}^N \alpha_i y_ix_i \\
	b &= \sum\limits_{i=1}^N \alpha_i y_i 
\end{align}
$$
$n_i$ 又称为第$i$个实例点更新的次数，实例点更新次数越多说明该点越接近超平面，也就越难正确分类。

### 1.4.2 算法
**输入**: 训练数据集$T = \{(x_1， y_1),(x_2,y_2),...,(x_N,y_N)\}$， 其中$x_i\in \mathcal{X}=R_n, y_i\in \mathcal{Y}=\{-1,+1\}, $$ i=1,2,...,N$; 学习率 $\eta$ $(0<\eta \leq1)$。
**输出**: $\alpha,b$；感知机模型$f(x)＝\text{sign}\left( \sum\limits_{j=1}^N \alpha_j y_j x_j \cdot x+b \right)$。 其中 $\alpha = (\alpha_1, \alpha_2, ..., \alpha_N)^T$
(1) $\alpha = 0, b = 0$
(2) 在训练集中选取数据$(x_i, y_i)$
(3) 如果 $y_i \left( \sum\limits_{j=1}^N \alpha_j y_j x_j \cdot x+b \right) \leq 0$:
$$
\begin{align}
\alpha_i &= \alpha_i +\eta\\
b &= b + \eta y_i  
\end{align}
$$
(4) 转至（2）直到没有误分类数据。
对偶形式中训练实例仅以内积的形式出现。 为了方便， 可以预先将训练集中实例间的内积计算出来并以矩阵的形式存储， 这个矩阵就是所谓的Gram矩阵（Gram matrix）
$$
G = [x_i \cdot x_j]_{N \times N}
$$


### 1.4.3 例子
数据同例1， 正样本点是$x_1＝(3,3)^T， x_2＝(4,3)^T$， 负样本点是$x_3＝(1,1)^T$， 试用感知机学习算法对偶形式求感知机模型.
- 代码
```Python
import numpy as np

# 变量初始化
x = np.matrix([[3, 3], [4, 3], [1, 1]])
y = np.matrix([[1], [1], [-1]])
alpha = np.matrix([0, 0, 0])
b = 0
lr = 1

# 计算Gamma 矩阵
G = np.matmul(x, x.T)

# 迭代更新 alpha 和 b
num = 0
while num<3:
    num = 0
    for i in range(len(x)):
        tmp = np.sum(np.multiply(np.multiply(alpha, y.T), G[i] ))
        tmp += b
        tmp *= y[i, 0]
        if tmp <= 0:
            alpha[0, i] += lr
            b += y[i]*lr
            print(alpha, b)
        else:
            num += 1
            
w = np.multiply(np.multiply(alpha.T, y), x)
w = np.sum(w, 0)
print("w:", w, " b:", b)
```
- 结果
```
[[1 0 0]] [[1]]
[[1 0 1]] [[0]]
[[1 0 2]] [[-1]]
[[1 0 3]] [[-2]]
[[2 0 3]] [[-1]]
[[2 0 4]] [[-2]]
[[2 0 5]] [[-3]]
w: [[1 1]]  b: [[-3]]
```

