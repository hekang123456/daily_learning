# Negative Sampling
### 1. 为什么要采用 Negative Sampling?
- 符号说明：
    |符号|含义|
    |----|----|
    |$n$|词表的大小|
    |$m$|词向量的长度|
    |$x_n$| $n$ 维的 one-hot 表示的词向量|
- 个人的理解：
原始的 word2vec 模型可以写成 $\text{Softmax}(x_n \times W_{n \times m} \times W_{m \times n})​$。从输入层到输出层的变换 $x_n \times W_{n \times m}​$ 可以仅仅看做是一个查表的过程，因为 $x_n​$ 是 one-hot 的表示所以并不费时间。 在原始的 word2vec 中从隐藏层到输出层的变化 $\hat{x_m} \times W_{m\times n}​$ 需要对 $W_{m \times n}​$ 中的每一个参数做运算。 同样的在  

