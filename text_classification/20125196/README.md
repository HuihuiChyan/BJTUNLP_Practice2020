1 数据处理

文本大小写转换、处理缩写词、去停用词、建立词表

文本切割和填充、为单词建立索引、转换索引为向量

将数据包装为Dataset类、传入DataLoader中

2 模型参数

embedding_size = 512；hidden_size = 1024；batch_size = 64；

input_channel = 1；output_channel = 512；dropout = 0.5；

output_size = 2；kernal_sizes = [2,3,4,5]；

3 优化器参数

learning_rate = 1e-4

4 模型

测试：model.eval()

模型存储：torch.save(model,'test.pth')

5 结果

最佳训练结果：train_acc = 0.998880；

最佳验证结果：test_acc = 0.881120；

6 注意

建立词表时，添加'[PAD]'；

建立词表时，添加'[unk]'；

dataloader本质是一个可迭代对象，循环DataLoader 对象，将数据加载到模型中进行训练，shuffle = True / False 需要注意
