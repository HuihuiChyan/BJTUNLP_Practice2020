数据处理：

- 使用torchtext将数据处理为可迭代的形式

词向量：

- 使用glove的300维预训练词向量：glove.6B.300d.txt

模型:

- 参考官网源码以及一些博客，将获取特征和计算得分等相关的函数改为batch形式，

- 只是用了word embedding，没有使用char级别的embedding

参数没怎么调，调了几次结果没有提高

