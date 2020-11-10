## 1 数据处理

##### 转换大小写、空格切割、torchtext生成data_iter

因为涉及到两个句子以及对应标签的同步，本来想沿用pack_padded_sequence，但是要考虑到两个句子不好同时输入网络，又得要相同的长度，所以改用torchtext来同时处理三个Field（sen1，sen2，label）,网络中加上mask操作来避免填充部分不必要的计算

----

## 2 ESIM

**1. Input Encoding**
Input Encoding实际上就是一层BiLSTM，论文中的$\overline{ai}$和$\overline{bj}$对应的是seq1和seq2每个time_step的特征输出

**2.  Local Inference Modeling**
- **Locality of inference**
  将上面BiLSTM的输出视为嵌入向量，并分别计算句子间的注意以使每个单词与假设的内容进行对齐，也就是通过对seq1和seq2的每一个time_step做内积，对应代码中attention函数中e~ij~的生成
- **Local inference collected over sequences**
  通过e~ij~获得seq1和seq2的相关性，e~ij~矩阵对列进行求和并获得概率乘以b~j~意思就是通过b~j~的加权和得到$\widetilde{ai}$，同理e~ij~矩阵对行进行求和并获得概率乘以a~i~意思就是通过a~i~的加权和得到$\widetilde{bj}$
- **Enhancement of local inference information**
  将Input Encoding得到的输出和上个步骤得到的输出进行差值计算和按元素乘积计算，然后与原始向量连接，也就是m~a~ = [ $\overline{a}$ ; $\widetilde{a}$ ; $\overline{a}$-$\widetilde{a}$ ; $\overline{a}$x$\widetilde{a}$]，同理m~b~ = [ $\overline{b}$ ; $\widetilde{b}$ ; $\overline{b}$-$\widetilde{b}$ ; $\overline{b}$x$\widetilde{b}$]，对应代码中ma和mb的计算
- **Pooling**
  通过计算平均池和最大池，并连接以形成最终的固定长度向量

**3.  BiLSTM和FC**
输出的时候再连接一层BiLSTM和FC层就可以输出了


----

该实验比较简单，理解起来也十分容易，着重在于复现了。数据集标签中的有‘-’的那一列还是去掉比较好，最终结果应该是划分为三类，我在代码中其实划分的是四类。