import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self,args,weight):
        super(TextCNN,self).__init__()
        # 超参数
        self.embedding_size = args.embedding_size
        self.vocab_size = args.vocab_size
        self.input_channel = args.input_channel
        self.output_channel = args.output_channel
        self.kernal_sizes = args.kernal_sizes
        self.output_size = args.output_size
        self.drop_out = args.dropout

        # 词向量层
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        # self.embedding.weight.data.copy_(weight) #使用预训练词向量
        # #self.embedding = self.embedding.from_pretrained(weight) # 使用预训练词向量，微调
        # self.embedding.weight.requires_grad = False # 不更新词向量参数，不进行微调
        
        # 卷积层  3层卷积，每层100个卷积核，每层的卷积核大小分别为3*3，4*4，5*5
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_channel, self.output_channel, (k, self.embedding_size)) for k in self.kernal_sizes]
        )

        # dropout层
        self.dropout = nn.Dropout(self.drop_out)
        # 全连接层
        self.linear = nn.Linear(len(self.kernal_sizes)*self.output_channel, self.output_size)

    def forward(self,x):
        # embedding层输出：(batch_size, max_len, embedding_size) 
        x = self.embedding(x) 

        # 添加输入通道数1，(batch_size, in_channel=1, max_len, embedding_size) 
        x = x.unsqueeze(1)

        # 卷积层输出:(batch_size, output_channel, max_len-k+1 , 1)
        # 降维后：(batch_size, output_channel, max_len-k+1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]

        # 最大池化层输出:(batch_size, output_channel, 1)
        # 降维后：(batch_size, output_channel)
        x = [F.max_pool1d(item,item.size(2)).squeeze(2) for item in x]

        # 组合不同卷积核提取的特征: (batch_size, len(kernal_sizes)*output_channel)
        x = torch.cat(x,1)

        x = self.dropout(x)

        # 全连接层输出
        out = self.linear(x)
        return out