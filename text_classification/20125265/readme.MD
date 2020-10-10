1、数据处理
替换缩略词、去掉停用词、将动词变回原型、去除所有非字母符号

2、TextCNN模型参数
filter_num = 512
filter_size = [2, 3, 4]
batch_size = 100
sen_len = 300
num_epoches = 20
lr = 0.001

3、使用环境
pytorch_GPU 1.2

注：使用的预训练词向量为GoogleNews-vectors-negative300.bin
