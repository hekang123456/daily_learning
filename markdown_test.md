---
title: 这是一篇对 hexo 中对 markdown 中各种格式支持的实验
date: 2018-11-25
categories: others
---



# 这是一篇对 hexo 中对各种格式支持的实验

- 添加图片

<img src="markdown_test/wallpaper.jpg">

- 公式
  - 插入公式 $\frac{1}{2}=0.5$  ,
    $$
      \begin{align}
      y &= \sigma (W[x,y]+b)\\
      x &= 0
      \end{align}
    $$









- 插入表格

| 表头1|表头2|表头3|表头4|
|-| :- | :-: | -: |
|默认左对齐|左对齐|居中对其|右对齐|
|默认左对齐|左对齐|居中对其|右对齐|
|默认左对齐|左对齐|居中对其|右对齐|

- 高亮：==哈哈哈哈==

- 删除线：~~哈哈哈~~

- 代码：

  ```python
  import tensorflow as tf
  if __name__ == "__main__":
      sum = 0
      for i in range(101, 200):
          flag = True
          for j in range(2，i//2+1:
              if i%j == 0:
                  flag = False
                  break
           if flag:
              sum += 1
       print(sum)
  ```


- [^]: 这是一段脚注

- > - 这是一段引用
  > - a
  > - b

- 

$\mathcal{X}, \mathcal{Y}$

 $$ L(Y,f(X)) = \left\{ \begin{align} 1&, Y\neq f(X) \\ 2&,Y=f(X)  \end{align} \right\}​$$
