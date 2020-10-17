## 1 数据处理

##### 转换大小写、空格切割、使用低频词和填充词、使用pad_sequence和pack_padded_sequence和pad_packed_sequence处理非定长序列

----

## 2 BiLSTM

##### 在处理BiLSTM的输入时使用pack_padded_sequence，输出使用pad_packed_sequence可以避免很多不必要的计算
##### 【注意：在使用这两个函数前要在DataLoader中按照seq_len对batch进行逆序处理】

----

## 3 CRF

#### loss部分 *（具体可看CRF_loss.jpg）*
##### 开源代码：先对一个time_step中的各个label做计算（内层for循环for next_tag in range(self.tagset_size):），再对每个time_step做计算（外层for循环for feat in feats:），并且开源代码中只是计算了一个句子，如果我们要增加batch的信息的话，就相当于我们要嵌套3层for循环。假设我们的输入为[batch_size, seq_lenth, output_size] = [512, 100, 11]，512个句子为一个batch_size，每个句子有100个词，每个词有对应的label维度是11，那么我们需要计算512x100x11次，这也就是为什么使用开源代码跑一个epoch能跑一个多两个小时的原因

##### 改进：在看log_sum_exp的时候，发现其实pytorch就有类似函数，即torch.logsumexp，所以就想着将循环计算改成矩阵计算（速度不要提升得太快）。
- 首先引入batch信息，对各个batch同时做计算。
- 然后发现在每个time_step中，forward_var（前面的分数）对于每个label计算是不变的，所以可以在label维度上进行扩展（emmm，复制）。因为引入了batch，所以也需要在batch维度上进行扩展。
- trainsitions转移矩阵在每个time_step上也是不变的，所以也需要在batch维度上进行扩展。
- 特征矩阵feat在每个label上都会重复计算，所以需要在forward_var上进行扩展。因为引入了batch，所以也需要在batch维度上进行扩展。
- 这样就变成了三个三维矩阵的相加，也就是转移矩阵transitions、前面time_step的分数矩阵forward_var和特征矩阵feat的相加（说白了就是找规律），就一次性对每个time_step同时运算了batch维度和label维度，将原来的512x100x11次循环运算变成了100次time_step的运算。
- 至于为什么最后的分数是logsumexp(每条路径)，可看代码中loss部分的注释（举了个例子）

#### forword部分 *（具体可看CRF_forward.jpg）*
##### 开源代码：同loss部分

##### 改进：也是将循环计算改成矩阵计算，但是loss部分只需要计算总体的路径分数，而forward阶段需要计算每个time_step的最大分数作为路径点，并且当前time_step的forward_var加上emit_score后还要作为下个time_step的初始forward_var，所以需要在计算的时候记录下路径和路径对应的分数（torch.max就可以得到）。

---