#### SVM中需要用到的相关的概念

<font color=red>欧式空间， 希尔伯特空间， 核函数， 凸优化问题</font>

- 仿射函数

  

- 凸优化问题
  $$
  \begin{align}
  \min_w &\quad f(w) \\
  s.t.  & g_i(w) \leq 0, i=1,2,...,k \\
  &h_i (w) =0, i=1,2,...,k \\
  \end{align}
  $$
  目标函数 $f(w)​$ , 约束条件 $g_i (w)​$ 都是 $R^n​$ 上的连续可微的凸函数，约束条件 $h_i (w)​$ 是 $R^n​$ 上的仿射函数。 则该优化问题为凸优化问题。

- 凸二次规划问题
  $$
  \begin{align}
   \min_w &\quad f(w) \\
   s.t.  \quad & g_i(w) \leq 0, i=1,2,...,k \\
   \quad &h_i (w) =0, i=1,2,...,k \\
  \end{align}
  $$
  

  目标函数 $f(w)$ 是二次函数, 约束条件 $g_i (w)$ 是仿射变换时，上诉凸优化问题成为凸二次规划问题。

 