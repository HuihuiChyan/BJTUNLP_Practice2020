数据处理：

- 低频词处理

词向量：

- 使用glove的300维预训练词向量

模型参数：

- 第一版：初始lr=0.001，大概前10轮会过拟合
  - lr：0.0001，max_len:400，output_channel:100，kernal_size:[2,3,4,5]，epoch：20
- 第二版，增加了句长max_len=2048、卷积通道数output_channel=512，以及迭代步数epoch=100
