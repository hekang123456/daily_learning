# 支持向量机（SVM）

#### 分类

线性可分支持向量机， 线性支持向量机，非线性支持向量机。

#### 选择不同支持向量机的情况

- 线性可分：  硬间隔SVM（线性可分SVM）
- 近似线性可分： 软间隔SVM。
- 线性不可分：核技巧， 软间隔最大化。



(>^<) (>^<) (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  

​															分割线

(>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<)  (>^<) (>^<) 



#### 线性可分SVM和硬间隔最大化

##### 1. 间隔

- 函数间隔：  $ \hat{\gamma}_i = y_i (w \cdot x_i + b) ​$ 分类正确，符号一致为正，否则为负， 分类正确的时候两者越接近越大，所以，我们希望函数间隔越大越好。
- 几何间隔： $ \gamma_i = y_i \left(\frac{w}{||w||} \cdot x_i + \frac{b}{||w||}\right)$ 
- 几何间隔和函数间隔：  如果  $w​$ 和 $b​$ 成比例的改变， 函数间隔也会成比例的改变， 几何间隔不变。
- 注： 带 $\hat{}​$ 表示的是函数间隔，不带 $\hat{}​$ 的表示几何间隔。

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
根据目标函数的等价性 $\max\limits_{w,b} \frac{1}{||w||} <==> \min\limits_{w,b} ||w|| <==>  \min\limits_{w,b} ||w||^2<==>  \min\limits_{w,b} \frac{1}{2}||w||^2​$ 可得：
$$
\begin{align}
& \min\limits_{w,b} \frac{1}{2}||w||^2 \\
&s.t. \quad y_i \left( w \cdot x_i  + b \right) \geq 1, i=1,2,...,N
\end{align}
$$




