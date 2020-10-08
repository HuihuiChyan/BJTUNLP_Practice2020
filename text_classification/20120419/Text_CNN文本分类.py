#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch as tc
from torch import nn
from torch.nn import functional as F
import pandas as pd
from torchtext import data
from torchtext.vocab import Vectors
import random
import time


# In[2]:


def get_test_data():
    val_data = pd.read_csv('./Dataset/test.csv')
    data = []
    for content, label in zip(val_data['content'], val_data['label']):
        data.append((content, label))
    random.shuffle(data)
    contents, True_labels = [],[]
    for content, label in data:
        contents.append(content)
        True_labels.append(label)
    return contents, True_labels

def get_dataset(csv_data, content_field, label_field, test= False):
    fields = [('None',None),('content',content_field),('label',label_field)]
    examples = []
    if test:
        for text in csv_data:
            examples.append(data.Example.fromlist([None, text, None], fields))
    else:
        for text, label in zip(csv_data['content'],csv_data['label']):
            examples.append(data.Example.fromlist([None,text,label], fields))
    return examples, fields
train_data = pd.read_csv('./Dataset/train.csv')
val_data = pd.read_csv('./Dataset/test.csv')
test_data = pd.read_csv('./Dataset/finnal_test.csv')
content = data.Field(sequential=True,lower=True, use_vocab=True)
label = data.Field(sequential=False, use_vocab= False)

# ------------------the way of get dataset----------------------
contents, True_labels = get_test_data()
# 获取构建Dataset所需的examples和fields
train_examples, train_fields = get_dataset(train_data, content, label)
val_examples, val_fields = get_dataset(val_data, content, label)
# 构建Dataset数据集
train_dataset = data.Dataset(train_examples, train_fields)
val_dataset = data.Dataset(val_examples, val_fields)
# 构建词典
content.build_vocab(train_dataset, vectors=Vectors(name ='./Dataset/glove.840B.300d.txt'))
# 构建数据迭代器
train_iter = data.BucketIterator(train_dataset, batch_size = 64, device = -1, shuffle = True, sort_key=lambda x:len(x.content))
val_iter = data.Iterator(val_dataset, batch_size = 64, device = -1, shuffle = True, sort = False, repeat= False)


# In[3]:


# 定义一维全局池化层
class GlobalMaxpool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxpool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size = x.shape[-1])

# 定义模型
class TextCNN(nn.Module):
    def __init__(self, vocab,embed_size, kernel_sizes, num_channels,output_size, device):
        super(TextCNN,self).__init__()
        self.Static_embedding = nn.Embedding(len(vocab), embed_size)
#         self.Non_static_embedding = nn.Embedding(len(vocab), embed_size)
        self.convs = nn.ModuleList()
        self.device = device
        for c,k in zip(num_channels, kernel_sizes): # 获取不同步长间的特征信息
            self.convs.append(nn.Conv2d(1, c, kernel_size = (k, embed_size)))
        self.maxpool1d = GlobalMaxpool1d() # 定义一维的全局池化层
        self.dropout = nn.Dropout(0.5) # 全连接层的输入进行dropout操作
        self.decoder = nn.Sequential(
            nn.Linear(sum(num_channels) * 10, 2),
        )
    # 定义k-max pooling
    def _fold_k_max_pooling(self, x, dim , k):
        # x.shape: [batch, c, seq_len']
        index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
        return x.gather(dim, index)

    def forward(self, x):
        # x.size(): (seq_len, batch)
        x = x.permute(1,0) # 调整回来
#         inputs = tc.cat((self.Static_embedding(x), self.Non_static_embedding(x)), dim = 2) # # (seq_len, batch, 2*embed_size)
        inputs = self.Static_embedding(x)
        inputs = inputs.permute(1,0,2)# (batch,seq_len,embed_size)
        inputs = inputs.unsqueeze(1) # (batch,1,seq_len,embed_size)
        # 使用max_pooling
#         encoding = tc.cat([self.maxpool1d(F.relu(conv(inputs).squeeze(-1))).squeeze(-1) for conv in self.convs ], dim = 1 )
        # 使用k_max_pooling
        encoding = tc.cat([self._fold_k_max_pooling(F.relu(conv(inputs).squeeze(-1)),dim= 2, k = 10).view(inputs.shape[0], -1) for conv in self.convs ], dim = 1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


# In[4]:


#验证集的准确率
def val_test(net, data_iter, loss_fc):
    acc_sum, loss_sum, n ,batch_count = 0.0, 0.0, 0, 0
    for batch in data_iter:
        X = batch.content
        y = batch.label
        X = X.to(device)
        y = y.to(device)
        X = X.permute(1,0)
        net.eval() # 进行评估模式
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
        loss_sum += loss_fc(y_hat, y).cpu().item()
        net.train() # 更改为训练模式
    return acc_sum / n, loss_sum/ batch_count

def train(net, device, train_iter, val_iter, num_epochs, loss_fc, optimizer):
    min_num =  0.8
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum,  n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for batch in train_iter:
            X = batch.content
            y = batch.label
            X = X.to(device)
            y = y.to(device)
            X = X.permute(1,0) #调整为(batch_size, seq_Len)，为了可以并行gpu计算
            y_hat = net(X)
            l = loss_fc(y_hat, y)
            optimizer.zero_grad() 
            l.backward()
            optimizer.step()
            train_l_sum += l
            batch_count += 1
            n += y.shape[0]
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()
        val_acc, val_num = val_test(net, val_iter, loss_fc)
        if val_acc >= min_num:
            min_num = val_acc
            print('Save model...')
            tc.save(net.state_dict() ,'./Model/model_normal.pth')
        print('epoch: %d, train_loss: %.3f train_acc: %.3f test_loss: %.3f val_acc: %.3f Time: %.3F'%(epoch, train_l_sum / batch_count, train_acc_sum /n, val_num, val_acc, time.time()-start))


# In[5]:


# 获取数据迭代器
device = tc.device('cuda:0')
# 定义模型
vocab, embed_size, kernel_sizes, num_channels,output_size = content.vocab, 300, [2,3,4,5], [400,400,400,400], 2
net = TextCNN(vocab,embed_size, kernel_sizes, num_channels,output_size, device) 
# 模型参数初始化
net.Static_embedding.weight.data.copy_(content.vocab.vectors)
# net.Non_static_embedding.weight.data.copy_(content.vocab.vectors)
net.Static_embedding.weight.requires_grad = False 
# 定义激活函数
lr, num_epochs = 0.0001, 100 
optim = tc.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr = lr)
loss = nn.CrossEntropyLoss() # 交叉熵损失函数
print(f'training on.. {device}')
net = nn.DataParallel(net, device_ids = [0,1]) 
net = net.to(device)
train(net, device, train_iter, val_iter, num_epochs, loss, optim) 


# In[ ]:




