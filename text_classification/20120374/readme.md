

# 第一版 acc=89.52%

```
batch_size =64
embedding_dim = 300
n_filters = 256
filters_sizes = [2,3,4,5]
sentence_max_len = 400
output_dim=2
dropout=0.5
num_epochs = 50
lr = 0.0001
```

# 第二版

改变模型结构：embedding层后加入bilstm层，首先将glove词向量送入bilstm层，然后输出的hidden与embedding合并（[batch_size,max_len,embedding_size+2*hidden_size]），接着做textcnn。

```
batch_size =64
embedding_dim = 300
hidden_size = 128
n_filters = 200
filters_sizes = [2,3,4,5]
sentence_max_len = 400
output_dim=2
dropout=0.5
num_epochs = 50
lr = 0.0001
```