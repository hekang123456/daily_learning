---
title: 贝叶斯公式+最大似然估计+最大后验概率公式+贝叶斯估计
categories: ML
tags: [贝叶斯公式， Bayes’ Theorem, 最大似然估计， 最大后验概率估计， MLE, MAP, maximum likelihood estimation, maximum a posterior probability estimation]
date: 2018-11-26
---
> 来源： https://blog.csdn.net/u011508640/article/details/7281598
# 贝叶斯公式+最大似然估计(MLE)+最大后验概率公式(MAP)+贝叶斯估计
## 1.贝叶斯公式
$$ P(A|B) = \frac{ P(B|A) \times P(A) }{ P(B|A) \times P(A) + P(B|\sim A) \times P(\sim A) } $$
- 作用： 
  - 你有多大把握相信一件证据。 给定 $B$ 的时候，你有多大的可能性会去相信 $A$ 能够成立。
  - 在做判断的时候需要考虑所有的因素。
    - 一件很难发生的事情 $P(A)$ 即使出现某个证据 $B$ 和它强相关 $P(B|A)$ 也要谨慎，因为证据可能来自其他虽然不是强相关但发生概率较高的事情 因为 $P(B|\sim A) \times P(\sim A)$ 可能会比较大从而导致$P(B|A)$ 比较小。 
  - 根据已知的或者主观容易断定的条件概率事件，计算出未知的或者较难评估的条件概率事件
  
## 2. 似然函数
对于函数 $P(x| \theta)$:
- 当 $\theta$ 是已知的情况下， $x$ 是变量， 这个函数叫做概率函数（probability function）, 用来描述对于不同的样本点 $x$ , 其出现的概率是多少。
- 当 $x$ 是已知的情况下， $\theta$ 是变量， 这个函数叫做似然函数（likelihood function）, **用来描述对于不同的模型参数**， 这个样本点出现的概率是多少。 

## 3. 最大似然估计（maximum likelihood estimation : MLE）
- **最大似然估计的核心思想是认为当前发生的事件是概率最大的**
- 构造一个关于参数 $\theta$ 的函数， 这个函数用来表示在已知的一组实验中产生了一组实验数据 $x_0$ 的可能性。
  - 在抛硬币实验中，每次抛硬币出现正反的概率满足二项分布。
  - 比如抛了10次，出现的一组实验数据 $x_0=[0111101110]$。 
  - 似然函数为： $f(\theta) = ((1−\theta) × \theta × \theta × \theta × \theta × (1 − \theta)× \theta × \theta × \theta ×(1−\theta))=\theta^7 \times (1 - \theta)^3$
- 计算使似然函数最大的参数值， 一般先取对数然后计算。
  - $\log f(\theta) = 7\log \theta + 3\log (1-\theta)$ 
  - 求导可以得到： $\frac{7-10\theta}{\theta (1-\theta)}$
  - 可以得到当$\theta = 0.7$ 的时候能够得到最大值。
## 4. 最大后验概率估计（maximum a posterior probability estimation: MAP）
最大似然估计的目的是通过求解得到 $\theta$ 使得似然函数 $P(x_0|\theta)$ 达到最大。 而最大后验概率估计是在最大似然估计的情况下考虑先验概率分布$P(\theta)$ 。使得 $P(\theta) \times P(x_0 | \theta)$ 达到最大。 
- 最大后验概率估计的目的其实是为了最大化后验： $P(\theta | x_0) = \frac{ P(x_0|\theta) \times P(\theta) }{P(x_0)} $ 因为 $P(x_0)$ 是可以通过做实验得到的。 所以只需要求解 $P(\theta) \times P(x_0 | \theta)$  使其最大。
- 最大后验的名字来源于  $P(\theta | x_0)$ 就是要在已有实验数据的情况下求解最合理的参数。

## 5. 贝叶斯估计 （感觉这部分理解的不是特别到位）
- 介绍： 不论是极大似然估计还是最大后验分布，我们都是通过构造一个似然函数（最大后验分布中还需要假设先验分布）， 来构建一个模型， 最后利用这个对数似然函数作为损失函数来求解相应的参数， 当参数固定的时候。模型也就确定了。  贝叶斯估计和最大后验相比目的不是为了得到一个最可靠的参数值，而是假设这个参数也服从某些分布。 因此我们需要通过一定的方法求解这个分布。
- 贝叶斯公式中$P(\theta | X) =  \frac{P(X | \theta) \times P(\theta)}{P(X)}$ 在最大后验估计和最大似然中有一个基本的假设是认为当前发生的事件概率最大， 通过带入具体的 $X=x_0$ 来求解参数。因此可以不用考虑 分母， 因为当$X=x0$ 的时候分母就是一个常数了。 <font color="#dd0000">但是现在我们希望求的是 $P(\theta | X)$ 这个分布， 因此 $P(X)$ 是不能够当成常数处理的， 这是一个分布。 $P(X)$ 可以通过联合概率进行求解：$\int_{\theta} P(X | \theta) d\theta$ . : 对于$P(X)$ 要怎么求 我还不确定， 搞懂了再补吧!</font>
  - 注： $P(X)$ 应该是通过实验作出来的， 当训练数据集确定了， 那么这个分布是能够确定下来的。
- 简便求解的方法： 将先验分布和后验分布构造成共轭先验。 那么可以将$P(X)$ 看成一个常数。因为这样知道了后验分布通过积分为1容易求得均值方差。


## 6. 一个简单的例子
投硬币10次得到的结果是$x_0 = [0111101110]$
- 最大似然函数， 上面已经说过了对应的似然函数是：  $f(\theta) =\theta^7 \times (1 - \theta)^3$
  - 代码：
  ```Python
  import math
  import matplotlib.pyplot as plt
  def mle_value():
      """最大似然估计： x表示 θ 值"""
      x = [0.001*i for i in range(0, 1000)]  # 不同的参数 θ 的值
      y = [i**7 * (1-i)**3 for i in x]  # θ对应的似然函数值

      print('对应最大值的θ是:', x[y.index(max(y))])

      plt.plot(x, y)
      plt.xlabel('θ')
      plt.ylabel('likelihood function value')
      plt.show()
  ```
  - 结果
  <img src="BT_MLE_MAP/1.png" width=400>
- 根据先验知识假定 P(θ) 为均值为0.5， 方差为0.1 的高斯函数，可以画出对应的概率密度图"
  - 代码
  ```Python
  def prior_value():
      """根据先验知识假定 P(θ) 为均值为0.5， 方差为0.1 的高斯函数，所以可以画出 θ 和 P(θ) 的图像： 一个高斯分布的密度函数，密度越大可能性越大"""
      def p_theta(u):
          return 1/((2*math.pi*0.01)**(1/2))*math.exp(-(u-0.5)**2/(2*0.01))
      x = [i*0.001 for i in range(0, 1000)]
      y = [p_theta(i) for i in x]

      print('对应最大概率密度的θ值是:', x[y.index(max(y))])

      plt.plot(x, y)
      plt.xlabel('θ')
      plt.ylabel("p(θ)")
      plt.show()
  ```
  - 结果
    <img src="BT_MLE_MAP/prior_value.png" width=400>
- $P(\theta)$ 的先验知识和似然函数$P(x_0 | \theta)$ 可以画出后验的图
  - 代码 
  ```Python
   """假定 p(θ) 满足均值为 0.5 方差为 0.1 的概率密度的情况下， 计算联合概率密度的值 p(xo|θ)*p(θ)， 联合概率反映了后验概率的数值大小 """
    def p_theta(u):
        return (1/((2*math.pi*0.01)**(1/2))*math.exp(-(u-0.5)**2/(2*0.01))) * (u**7 *(1-u)**3)
    x = [i*0.001 for i in range(0, 1000)]
    y = [p_theta(i) for i in x]

    print('对应最大联合概率密度的θ值是:', x[y.index(max(y))])

    plt.plot(x, y)
    plt.xlabel('θ')
    plt.ylabel("p(xo|θ)*p(θ)")
    plt.show()
  ```
  - 结果
  <img src="BT_MLE_MAP/map_value.png" width=400>
- $P(\theta)$ 的先验知识和似然函数$P(x_0 | \theta)$ 通过多做几次实验可以得到更加准确的结果
  - 代码
  ```Python
  def map_value100():
    """ 实验了100次会得到的结果 """
    def p_theta(u):
        return (1/((2*math.pi*0.01)**(1/2))* math.exp(-(u-0.5)**2/(2*0.01))) * (u**7 *(1-u)**3)**70*(u**7 *(1-u)**3)**30

    x = [i*0.001 for i in range(0, 1000)]
    y = [p_theta(i) for i in x]

    print('对应最大联合概率密度的θ值是:', x[y.index(max(y))])

    plt.plot(x, y)
    plt.xlabel('θ')
    plt.ylabel("p(xo|θ)*p(θ)")
    plt.show()
  ```
  - 结果
  <img src="BT_MLE_MAP/map_value100.png" width=400>

  








