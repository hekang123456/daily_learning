# EM算法

- EM 算法是不断的求解**下界极大化逼近求解对数似然极大化**的方法。
- EM 算法不能保证求解得到全局最优解。

### 符号说明和关键词说明

|符号 |含义|
|----|----|
|$Y$| 观测随机变量|
|$Z$| 隐随机变量|
|$\theta$| 需要估计的模型参数|
|完全数据| 观测随机变量$Y$ 和隐随机变量 $Z$ 连在一起称为完全数据 |
|不完全数据| 观测数据 $Y$ 称为不完全数据 |

### 1. 作用

- 用于求解含有隐变量的概率模型的极大似然估计。
- 极大后验概率进行估计。

### 2. （Q）函数

​	**完全数据**的**对数似然**函数 $\log P(Y, Z|\theta)$ 关于在给定观测数据 $Y$ 和 当前参数 $\theta^{(i)}$ 下对未观测数据**$Z$ 的条件概率分布** $P(Z|Y, \theta^{(i)})$ 的**期望**称为 $Q$ 函数。

​	$$Q(\theta, \theta^{(i)}) = E_z [\log P(Y,Z|\theta) |Y, \theta^{(i)}]$$

### 3. （EM算法）

- 输入： 观测变量数据$Y$,  隐变量数据$Z$, 联合分布 $P(Y,Z |\theta)$ , 条件分布 $P(Z|Y, \theta)$ 。

- 输出：模型参数。

- 计算过程

  - 选择参数<font color=#CD00CD>初值 $\theta^{(0)}$  </font>开始迭代；

  - E步： 在当前 $\theta^{(i)}$ 的情况下，<font color=#CD00CD>计算 $Q(\theta, \theta^{(i)})$</font> ， 因为计算 $Q$ 函数就是在求期望，所以称为 $E$ 步。

    $$\begin{align} Q(\theta, \theta^{(i)}) &= E_z [\log P(Y,Z| \theta)| Y, \theta^{(i)}] \\ &=\sum_{z} \log P(Y,Z|\theta) P(Z|Y,\theta^{(i)}) \end{align}​$$ 

  - M步：求使 $Q$ 函数<font color=#CD00CD>最大的 $\theta$</font> 。 

    $$\theta^{(i+1)} = \arg \max\limits_{\theta} Q(\theta, \theta^{(i)})$$

  - 上面两步直到<font color=#CD00CD>收敛</font>。

### 4. 说明

- EM  算法的初值可以任意选择，但是EM算法对初值是敏感的。
- EM 算法每次迭代使似然函数增大或达到局部极值。
- 停止迭代条件： $||\theta^{(i+1)} - \theta^{(i)} || < \epsilon_1$ 或者 $|| Q(\theta, \theta^{(i+1)}) - Q(\theta, \theta^{(i)}) < \epsilon_2||$

### 5. 简单叙述 EM 算法的<font color=436EEE>核心思想</font> (由来)

- **函隐变量的概率模型无法直接求解最大对数似然**：对一个含有隐变量的概率模型，目标是极大化观测数据（不完全数据）$Y$ 关于参数 $\theta$ 的对数似然函数，即：

  $$L(\theta) = \log P(Y|\theta) = \log \sum_z P(Y,Z | \theta) = \log ( \sum_z P(Y|Z, \theta) P(Z|\theta))​$$

  <font color=red>要极大化 $L(\theta)$ 需要对 未知的隐变量 $Z$ 进行求和或者求积分，无法求解。</font>

- **迭代方法** ： 在每一步的迭代中 $L (\theta^{(i+1)}) > L(\theta^{(i)})$ 

- 具体怎么迭代？

  - 考虑 $L(\theta) - L(\theta^{(i)})$ 
  - 得到  $L(\theta)$ 的一个下界 $B(\theta, \theta^{(i)})$
  - 使得 下界最大， $B(\theta, \theta^{(i )})$ 最大，最后推出了最大化 $Q(\theta, \theta^{(i)})$

