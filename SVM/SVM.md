支持向量机（SVM）

#### 分类

线性可分支持向量机， 线性支持向量机，非线性支持向量机。

#### 选择不同支持向量机的情况

- 线性可分：  硬间隔SVM（线性可分SVM）
- 近似线性可分： 软间隔SVM。
- 线性不可分：核技巧， 软间隔最大化。



----------

​																		（>^<）这里有一条分割线（>^<）

----





#### (一). 线性可分SVM的导出

##### 1. 间隔

- 函数间隔：  $ \hat{\gamma}_i = y_i (w \cdot x_i + b) $ 分类正确，符号一致为正，否则为负， 分类正确的时候两者越接近越大，所以，我们希望函数间隔越大越好。
- 几何间隔： $ \gamma_i = y_i \left(\frac{w}{||w||} \cdot x_i + \frac{b}{||w||}\right)$ 
- 几何间隔和函数间隔：  如果  $w$ 和 $b$ 成比例的改变， 函数间隔也会成比例的改变， 几何间隔不变。
- 注： 带 $\hat{}$ 表示的是函数间隔，不带 $\hat{}$ 的表示几何间隔。

#####  2. 间隔最大化 ==> 约束最优化问题 

最小函数间隔最大化：
$$
\begin{align}
& \max\limits_{w,b} \min\limits_i \gamma_i \\
&s.t. \quad y_i \left( \frac{w}{||w||}x_i  + \frac{b}{||w||} \right) \geq \min\limits_i \gamma_i , i=1,2,...,N
\end{align}
$$
考虑几何间隔和函数间隔的关系可得：
$$
\begin{align}
& \max\limits_{w,b} \min\limits_i \frac{\hat{\gamma_i}}{||w||} \\
&s.t. \quad y_i \left( w \cdot x_i  + b \right) \geq \min\limits_i \hat{\gamma_i}, i=1,2,...,N
\end{align}
$$
当 $w$  和 $b$ 变按比例变为 $\lambda w$ 和 $\lambda b$ 时， <font color=red>目标函数和约束条件都没有变（分子分母都按比例变化，左右两边都按比例变化）</font>。 因此，我们可以取 $\min\limits_i \hat{\gamma_i} = 1$ 则最优化问题变为：
$$
\begin{align}
& \max\limits_{w,b} \frac{1}{||w||} \\
&s.t. \quad y_i \left( w \cdot x_i  + b \right) \geq 1, i=1,2,...,N
\end{align}
$$
根据目标函数的等价性 $\max\limits_{w,b} \frac{1}{||w||} <==> \min\limits_{w,b} ||w|| <==>  \min\limits_{w,b} ||w||^2<==>  \min\limits_{w,b} \frac{1}{2}||w||^2$ 可得：

$$
\begin{align}
& \min\limits_{w,b} \frac{1}{2}||w||^2 \\
&s.t. \quad y_i \left( w \cdot x_i  + b \right) \geq 1, i=1,2,...,N
\end{align}
$$


#### (二). 硬间隔最大化SVM的规划问题

$$
\begin{align}
& \min\limits_{w,b} \frac{1}{2}||w||^2 \\
&s.t. \quad y_i \left( w \cdot x_i  + b \right) \geq 1, i=1,2,...,N
\end{align}
$$



 ####  (三). 软间隔最大化SVM的规划问题

$$
\begin{align}
\min\limits_{w,b,\xi}\quad &  \frac{1}{2}||w||^2 + C\sum\limits_{i=1}^N \xi_i \\
s.t. \quad & y_i \left( w \cdot x_i  + b \right) \geq 1-\xi_i, i=1,2,...,N \\
&\xi_i \geq 0, i=1,2,...,N
\end{align}
$$

#### (四). SVM 的拉格朗日函数

- 拉格朗日函数
  $$
  \begin{align}
  L(w,b,\alpha) &= \frac{1}{2} ||w||^2 - \sum\limits_i \alpha_i (y_i (w \cdot x_i+b)-1) \\
  &= \frac{1}{2} ||w||^2 - \sum\limits_i \alpha_i y_i (w\cdot x_i +b) + \sum\limits_i \alpha_i
  \end{align}
  $$
  拉格朗日函数的作用： 将带有约束的优化问题转化为不带约束的极值问题。

- 拉格朗日函数和原始问题的关系; [出处](https://blog.csdn.net/LilyZJ/article/details/88778940)
  $$
  \max\limits_{\alpha, \beta;\beta_i \geq 0} L(x, \alpha, \beta) = 
  \left\{
  \begin{align}
  &f(x) , &x满足原始问题的约束 \\
  &+\infty , &其他
  \end{align}
  \right.
  $$
  上述公式描述的是，对于规划问题的广义拉格朗日函数，当 $x$ 满足原始问题的约束的情况下，其最大值就等于原始问题对应的结果。

#### (五). 原始问题的无约束形式和对偶问题

- 原始问题的无约束形式
  $$
  \min\limits_{w,b} \max\limits_{\alpha} L(w,b,\alpha)
  $$
  根据（四）中提到的拉格朗日函数和原始问题的关系可以得到，原始问题的无约束形式。

- 原始问题的对偶问题
  $$
  \max\limits_\alpha \min\limits_{w,b} L(w,b,\alpha)
  $$

#### (六). 通过对偶问题求解支持向量机

- 求 $\min\limits_{w,b} L(w,b,\alpha)$ 

  将拉格朗日函数 $L(w,b,\alpha)$ 分别对 $w$, $b$ 求偏导并令其等于0，得到两个等式：
  $$
  \begin{align}
  \sum\limits_{i=1}^N \alpha_i y_i x_i &=w\\
  \sum\limits_{i=1}^N \alpha_i y_i &= 0
  \end{align}
  $$
  

  将两个等式带入 拉格朗日函数中，去掉 $w$ 和 $b$ 得到：
  $$
  \min\limits_{w,b} L(w,b,\alpha) = -\frac{1}{2} \sum\limits_{i=1}^N \sum\limits_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) + \sum\limits_{i=1}^N \alpha_i
  $$

- 求 $\max\limits_\alpha \min\limits_{w,b} L(w,b,\alpha)$ 
  $$
  \begin{align}
  \max\limits_\alpha 
  \quad & -\frac{1}{2} \sum\limits_{i=1}^N \sum\limits_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) + \sum\limits_{i=1}^N \alpha_i \\
  s.t. \quad & \sum\limits_{i=1}^N \alpha_i y_i = 0 \\
  &\alpha_i \geq 0, i=1,2,...,N
  \end{align}
  $$
  ==>
  $$
  \begin{align}
  \min\limits_\alpha 
  \quad & \frac{1}{2} \sum\limits_{i=1}^N \sum\limits_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j) - \sum\limits_{i=1}^N \alpha_i \\
  s.t. \quad & \sum\limits_{i=1}^N \alpha_i y_i = 0 \\
  &\alpha_i \geq 0, i=1,2,...,N
  \end{align}
  $$

#### (七). 关于通过对偶方式求得的解是最优解的说明

- 定理1： 若原问题和对偶问题都有最优值，则：
  $$
  d^*= \max\limits_{\alpha,\beta； \alpha_i\geq0} \min\limits_x L(x, \alpha,\beta) \leq \min\limits_x\max\limits_{\alpha,\beta:\alpha_i \geq 0} L(x, \alpha, \beta) = p^*
  $$

- 推论1： 假设 $x^*, \alpha^*, \beta^*$ 分别是原始问题和对偶问题的可行解， 并且 $d^* = p^*$, 则 $x^*, \alpha^*, \beta^*$  分别是原始问题和对偶问题的最优解。

- 定理2： 在有约束的最优化问题中，如果 目标函数 $f(x)$, 不等式约束 $c_i (x)$ 是凸函数，等式约束 $h_i(x)$ 是仿射函数，并且假设不等式约束是严格可行的，即存在 $x$ ,对所有的 $i$ 有 $c_i (x) <0$, 则存在  $x*, \alpha^*, \beta^*$ 是原始问题的解也是对偶问题的解。 并且 $d^* = p^*$。

- 定理3：  在有约束的最优化问题中，如果 目标函数 $f(x)$, 不等式约束 $c_i (x)$ 是凸函数，等式约束 $h_i(x)$ 是仿射函数，并且假设不等式约束是严格可行的。则 $x^*, \alpha^*, \beta^*$ 分别是原始问题和对偶问题的解的充分必要条件是 $x^*, \alpha^*, \beta^*$ 满足 KKT 条件。
  $$
  \begin{align}
  \bigtriangledown _x L(x^*, \alpha^*, \beta^8) &=0 \\
  \alpha_i ^* c_i (x^*) &=0, i=1,2,...,k \\
  c_i(x^*) &\leq 0, i= 1,2,...,k\\
  \alpha_i^* &\geq 0, i=1,2,...,k\\
  h_j(x^*) &=0, j=1,2,...,l
  \end{align}
  $$


因为 SVM 中的目标函数和非线性约束条件是凸函数，等式约束条件是仿射函数，并且在《统计学习方法》P122 中证明了对偶的解满足 KKT 条件因此其满足定理2，3 并且根据推论1可知是最优解。

#### （八） 合页损失函数

线性支持向量机（硬间隔和软间隔）等价于合页损失函数加上正则化项。
$$
\min\limits_{w,b} \quad \sum\limits_{i=1}^N [1-y_i(w\cdot x_i +b)]_+ + \lambda ||w||^2 \\
[z]_+ = 
\left\{
\begin{align}
z, z>0 \\
0, z\leq 0
\end{align}
\right.
$$





----

​											                     （>^<）这里有一条分割线（>^<）

------



###  非线性支持向量机与核函数

#### （一） 核函数的定义

​	假设 $\mathcal{X}$ 是输入空间 （欧式空间 $R^n$ 的子集或离散集合），又设$\mathcal{H}$ 为特征空间（希尔伯特空间），如果存在一个从 $\mathcal{X}$ 到 $\mathcal{H}$ 的映射：
$$
\phi(x): \mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有的 $x,z\in \mathcal{X}$ ， 函数 $K(x,z)$ 满足条件：
$$
K(x,z) = \phi(x) \cdot \phi(z)
$$
则称 $K(x,z)$ 为核函数，$\phi (x)$ 为映射函数，式中 $\phi(x) \cdot \phi(z)$ 为 $\phi(x) $ 和 $\phi(z)$ 的内积。

#### （二）正定核的充要条件

设 $K: \mathcal{X } \times \mathcal{X} \rightarrow R$ 是对称函数， 则 $K(x,z)$ 为正定核函数的充要条件是对任意的 $x_i \in \mathcal{X}, i=1,2,...,m$ , $K(x,z)$ 对应的 Gram 矩阵：
$$
K = [K(x_i, x_j)]_{m \times m} 
$$
是半正定矩阵。

#### （三） 正定核的等价定义

设 $\mathcal{X} \subset R^n$, $K(x,z)$ 是定义在 $\mathcal{X} \times \mathcal{X}$ 上的对称函数，如果对任意 $x_i \in \mathcal{X}, i=1,2,...,m$ 。 $K(x,z)$ 对应的 Gram 矩阵
$$
K = [K(x_i, x_j)]_{m \times m}
$$
是半正定矩阵，则称 $K(x,z)$ 是正定核。

#### （四) 说明

- 通常所所的核函数就是正定核
- 虽然正定核的定义在构造核函数时很有用。 但对于一个具体函数 $K(x,z)$ 来说，检验它是否为正定核函数并不容易，因为要求对任意有限输入集 $\{x_1, x_2, ...,x_n\}$ 验证 $K$ 对应的 Gram 矩阵是否为半正定的。<font color=red>在实际问题中往往应用已有的核函数。</font>

#### (五) 常用的核函数

- 多项式核函数

$$
K(x, z) = (x\cdot z +1)^p
$$

​	分类决策函数为
$$
f(x) = \text{sign} \left( \sum\limits_{i=1}^{N_s} \alpha_i ^* y_i (x_i \cdot x +1)^p +b^* \right)
$$

- 高斯核函数
  $$
  K(x,z) = \exp \left( -\frac{||x-z||^2}{2 \sigma^2} \right)
  $$
   分类决策函数为
  $$
  f(x) = \text{sign} \left(\sum\limits_{i=1}^{N_s} \alpha_i^* y_i \exp\left( -\frac{||x-x_i||^2}{2\sigma^2} \right)  + b^* \right)
  $$

- 字符串核函数

  假设有两个字符串 $s$, $t$ 。 那么在这两个字符串上的核函数是基于映射 $\phi_n$ 的特征空间中的内积：
  $$
  \begin{align}
  k_n(s,t) &= \sum\limits_{u \in \Sigma^n } [\phi_n (s)]_u [\phi_n (t)]_u \\
  &= \sum\limits_{u \in \Sigma^n } \sum\limits_{(i,j): s(i)=t(j)=u} \lambda^{l(i)} \lambda^{l(j)}
  \end{align}
  $$
  $\Sigma^n$ 表示 特征空间的维度是$n$，每个维度都对应与一个字符串 $u$。 $l(i)$ 表示字符串 $i$ 的长度， $i$ 是$u$ 所处的字串的长度（$u$ 可以以不连续的形式出现）。例如 $[\phi_3 (lass\Box das )]_{asd} =2 \lambda^5$ ,  $[\phi_3(Nasdaq)]_{asd}=\lambda^3$ 。在第一个字符串中 $asd$ 是长度为5的不连续子串，共出现了2次。 在第一个字符串中 $asd$ 是连续的子串。 字符串核函数给出了字符串 $s$ 和 $t$ 中长度等于 $n$ 的所有字串组成的特征向量的余弦相似度。 直观上，两个字符串相同的字串越多，它们就越相似，字符串核函数的值就越大。 字符串核函数可以由动态规划快速地计算。

#### （六）<font color=red> 非线性支持向量机学习算法</font>

- 选取适当的核函数 $K(x,z)$ 和适当的参数 $C$ ，构造并求解最优化问题：
  $$
  \begin{align}
  \min\limits_\alpha  \quad& \frac{1}{2} \sum\limits_{i=1}^N \sum\limits_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i , x_j) - \sum\limits_{i=1}^N \alpha_i \\
  s.t. \quad& \sum\limits_{i=1}^N \alpha_i y_i = 0 \\
  & 0\leq \alpha_i \leq C, i=1,2,...,N\\
  \end{align}
  $$
  求得最优解 $\alpha^* = (\alpha_1^* , \alpha_2^*, ...,\alpha_N^*)^T$。

- 选择 $\alpha^*$ 的一个正分量 $0 < \alpha_j^* < C$, 计算：
  $$
  b^*= y_j - \sum\limits_{i=1}^N \alpha_i ^* y_i K(x_i, x_j)
  $$

- 构造决策函数
  $$
  f(x) = \text{sign} \left( \sum\limits_{i=1}^N \alpha_i^* y_iK(x, x_i)+b^* \right)
  $$
  当 $K(x,z)$ 是正定核函数时， 上面构造的最优化问题是凸二次规划问题，解是存在的。







------

​											                     （>^<）这里有一条分割线（>^<）

------

### 序列最小化优化算法

#### （一）SMO 算法

- 取初值 $\alpha^{(0)} = 0$, 令 $k=0$。
- 选取优化变量 $\alpha_1^{(k)}, \alpha_2^{(2)}$ ， 解析求解两个变量的最优化问题。 求的最优解 $\alpha_1^{(k+1)}$, $\alpha_2^{(k+1)}$ ， 更新 $\alpha$ 为 $\alpha^{(k+1)}$ 

- 若在精度 $\epsilon$ 范围内满足停机条件
  $$
  \sum\limits_{i=1}^N \alpha_i y_i =0, 0\leq \alpha_i \leq C, i=1,2,...,N \\
  y_i g(x_i) = \left\{
  \begin{align}
  \geq 1, &\{ x_i | \alpha_i =0 \} \\
  =1, &\{ x_i | 0<\alpha_i < C\} \\
  \leq 1,& \{ x_i | \alpha_i =C \} 
  \end{align}
  \right.
  $$
  其中，
  $$
  g(x_i) = \sum\limits_{j=1}^N \alpha_j y_j K(x_j , x_i ) + b
  $$
  则转 最后一步，否则令 $k=k+1$, 转第二步。

- 取  $\hat{\alpha} = \alpha^{（k+1）}$

#### （二）两个变量二次规划的求解方法

- 写出两个变量二次规划的表达式

- 根据 $y_1, y_2$ 符号是否相同确定 $\alpha_1, \alpha_2$ 的上下界。 $\alpha_1, \alpha_2$ 的取值在在与一个正方形的对角线平行的线段上。

- 计算最优化问题沿着约束方向不考虑约束范围的解是：
  $$
  \alpha_2^{new, unc} = \alpha_2^{old} + \frac{y_2(E_2- E_1)}{\eta}
  $$
  其中  
$$
  \eta = K_{11} + K_{22} - 2K_{12} = ||\Phi(x_1)-\Phi(x_2)||^2 \\
  E_i = g(x_i) -y_i = \left( \sum\limits_{j=1}^N \alpha_j y_j K(x_j ,x_i)+b \right) -y_i
  $$
  
  
- 对上一步的解，考虑第二步中计算得到的约束范围。
  $$
  \alpha_2^{new} = \left\{
  \begin{align}
  &H, & \alpha_2^{new, unc} > H \\
  &\alpha_2^{new, unc}, & L \leq \alpha_2^{new, unc} \leq H\\
  &L, & \alpha_2^{new, unc} \leq L
  \end{align}
  \right.
  $$

- 由  $\alpha_2^{new}$ 可以求的 $\alpha_1^{new}$ :
  $$
  \alpha_1 ^{new} = \alpha_1^{old} + y_1y_2(\alpha_2^{old} - \alpha_2^{new})
  $$

#### （三）变量的选择

- 第一个变量的选择

  选择违反 KKT 条件最严重的样本点。 对应的KKT 条件是：
  $$
  \begin{align}
  \alpha_i=0 &\Longleftrightarrow  y_i g(x_i) \geq 1 \\
  0 < \alpha_i <C &\Longleftrightarrow y_i g(x_i) = 1\\
  \alpha_i =C &\Longleftrightarrow y_i g(x_i) \leq 1
  \end{align}
  $$
  其中 $g(x_i) = \sum\limits_{j=1}^N \alpha_j y_j K(x_j , x_i)+b$。
  
  在检验过程中，外层循环首先遍历所有满足条件 $0<\alpha_i < C$ 的样本点。 如果都满足那么遍历整个训练集，检验他们是否满足 KKT 条件。

-  第二个变量的选择

  选择第二个变量称为内层循环。 选择 $|E_1 - E_2|$ 最大的那个变量。

- 计算 $b$ 和差值 $E_i$

  在每次完成两个变量的优化后，都要重新计算阈值 $b$.
  $$
  b_1^{new} = -E_1 -y_1K_{11} (\alpha_1^{new} -\alpha_1^{old}) - y_2K_{21} (\alpha_2^{new}-\alpha_2^{old}) + b^{old} \\
  b_2^{new} = -E_2 -y_1K_{12} (\alpha_1^{new} -\alpha_1^{old}) - y_2K_{22} (\alpha_2^{new}-\alpha_2^{old}) + b^{old}
  $$
  如果 $\alpha_1^{new}, \alpha_2^{new}$ 同时满足条件 $0<\alpha_i^{new}<C$ 那么$b_1^{new}, b_2^{new}$ 相等。 否则取 两者的中点作为 $b^{new}$。

  重新计算 $E_i$ 的值。
  $$
  E_i^{new} = \sum\limits_S y_j \alpha_j K(x_i ,x_j ) + b^{new} - y_i
  $$
  其中 $S$ 是所有支持向量的集合。

