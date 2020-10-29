# preprocess.py

将数据集中text和label转成csv格式，方便使用torchtext进行数据处理

# crf.py

官网提供的crf代码只能一句一句输入（很不友好），为了提高并行运算速度，改成了batch版本。

由于一个batch里每句话长短不同，进行loss计算时必须将padding mask掉。

# bilstm-crf.py 

 此版本只是复现了论文中bilstm-crf的结构。

模型优化时可以进一步引入char embedding ，通过cnn卷积后，与word embedding 拼接输入到bilstm中。

# main.py

利用torchtext生成dataIterator送入模型训练&预测。

不限制SENTENCE_MAX_LEN要比限制句子长度结果好很多，其他参数就随缘调的，结果不是很好，有点过拟合。







