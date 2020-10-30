#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch as tc
from torch import nn
import numpy as np
from torch import optim
import torchtext
from torchtext import data
from torchtext.vocab import Vectors
import argparse
import time
import torch.nn.functional as F


# In[2]:


# 定义超参数
parse = argparse.ArgumentParser()
parse.add_argument('--max_len', type=int, default = 100, help='文本截取长度')
parse.add_argument('--glove840_path',type=str, default='../TextClassification_based_on_textCNN/Dataset/glove.840B.300d.txt',help='词向量路径')
parse.add_argument('--batch_size', type=int, default = 32, help='一个批次训练样本个数')
parse.add_argument('--embed_size', type=int, default = 300, help='词向量维度')
parse.add_argument('--hidden_size', type=int, default = 300, help='隐藏层维度')
parse.add_argument('--output_size', type=int, default = 3, help='最终的标签Label数')
parse.add_argument('--lr', type=float, default = 0.0004 , help='学习率')
parse.add_argument('--num_epochs', type=int, default = 10 , help='训练迭代次数')
args = parse.parse_args(args = [])
label_to_idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


# In[3]:


# 处理数据
def get_dataiter(path, sequence, type = 'train'):
    lables, sentence1s, sentence2s = [], [], []
    #定义三个field
    label = data.Field(sequential = False, use_vocab = False)
    # 获取数据
    with open(path, 'r') as fp:
        dataset = fp.readlines()
    # 整理成examples
    examples = []
    fields = [('seq_1',sequence), ('seq_2',sequence), ('label',label)]
    for idx in range(1, len(dataset)):
        label = dataset[idx].split('\t')[0]
        seq_1 = dataset[idx].split('\t')[5]
        seq_2 = dataset[idx].split('\t')[6]
        if label == '-':
            continue
        examples.append(data.Example.fromlist([seq_1, seq_2 ,label_to_idx[label]], fields))
    # 构建Daraset数据集
    data_set = data.Dataset(examples, fields)
    print(len(data_set))
    # 如果是训练集则进行构造词典
    if type == 'train':
        sequence.build_vocab(data_set, vectors = Vectors(args.glove840_path))
        dataiter = data.BucketIterator(data_set, batch_size= args.batch_size, shuffle = True)
        return sequence, dataiter
    else:
        dataiter = data.BucketIterator(data_set, batch_size= args.batch_size, shuffle = False)
        return dataiter
    
train_path = './Dataset/snli_1.0_train.txt'
test_path = './Dataset/snli_1.0_test.txt'
val_path = './Dataset/snli_1.0_dev.txt'
sequence = data.Field(sequential= True, use_vocab = True, lower = True, fix_length = args.max_len, include_lengths = True,batch_first = True)
sequence, train_iter = get_dataiter(train_path, sequence, type= 'train')
test_iter = get_dataiter(test_path, sequence, type= 'test')
val_iter = get_dataiter(val_path, sequence, type= 'val')


# In[4]:


# 定义模型
class ESIM_model(nn.Module):
    def __init__(self, vocab):
        super(ESIM_model, self).__init__()
        self.embedding = nn.Embedding(len(vocab), args.embed_size)
        self.Bi_lstm1 = nn.LSTM(args.embed_size, args.hidden_size, bidirectional = True, batch_first = True)
        self.Bi_lstm2 = nn.LSTM(args.hidden_size, args.hidden_size, bidirectional = True, batch_first = True)
        self.linear1 = nn.Linear(args.hidden_size * 8, args.hidden_size)
        self.linear2 = nn.Linear(args.hidden_size, args.output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.5)
        
    def _attention(self, seq1_decoder, seq2_decoder, mask1, mask2):
        # seq_size: [batch, seq_len, hidden_size]
        scores_b = tc.matmul(seq1_decoder, seq2_decoder.permute(0,2,1))
        scores_a = tc.matmul(seq2_decoder, seq1_decoder.permute(0,2,1))
        # 进行对seq_1 & seq_2进行mask操作
        scores_b = scores_b.masked_fill(mask2.unsqueeze(-1) == 1, -1e9)
        scores_a = scores_a.masked_fill(mask1.unsqueeze(-1) == 1, -1e9)
        #归一化 [batch. seq_len, seq_len]，其中dim=1为a，dim=2为b
        scores_b = F.softmax(scores_b, dim = -1)
        scores_a = F.softmax(scores_a, dim = -1)
        seq1_decoder_attention = tc.matmul(scores_b, seq2_decoder)
        seq2_decoder_attention = tc.matmul(scores_a, seq1_decoder)
        return seq1_decoder_attention, seq2_decoder_attention
    
    def _get_max_avg(self, x):
        # x.size : [batch, seq_len, 2 * hidden_size]
        p1 = F.avg_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        return tc.cat([p1,p2], dim = -1)
        
    def forward(self, seq1, seq2, mask1, mask2):
        #嵌入层
        seq1_embedding = self.embedding(seq1)
        seq2_embedding = self.embedding(seq2)
        #第一层lstm+attention
        seq1_decoder, _ = self.Bi_lstm1(seq1_embedding)
        seq2_decoder, _ = self.Bi_lstm1(seq2_embedding)
        seq1_decoder_attention, seq2_decoder_attention = self._attention(seq1_decoder, seq2_decoder, mask1, mask2)
        #信息拼接
        m_seq1 = tc.cat((seq1_decoder,seq1_decoder_attention,seq1_decoder-seq1_decoder_attention,tc.mul(seq1_decoder,seq1_decoder_attention)), dim = -1)
        m_seq2 = tc.cat((seq2_decoder,seq2_decoder_attention,seq2_decoder-seq2_decoder_attention,tc.mul(seq2_decoder,seq2_decoder_attention)), dim = -1)
        #全连接层
        m_seq1_decoder =  self.relu(self.linear1(m_seq1))
        m_seq2_decoder =  self.relu(self.linear1(m_seq2))
        #第二层lstm
        f_seq1_decoder, _ = self.Bi_lstm2(m_seq1_decoder)
        f_seq2_decoder, _ = self.Bi_lstm2(m_seq2_decoder)
        #max+ave信息拼接
        seq1_max_avg = self._get_max_avg(f_seq1_decoder)
        seq2_max_avg = self._get_max_avg(f_seq2_decoder)
        f_x = tc.cat([seq1_max_avg, seq2_max_avg], dim = -1)
        logit = self.linear2(self.dropout(self.tanh(self.linear1(f_x))))
        return logit


# In[9]:


#验证集的准确率
def val_test(net, data_iter, loss_fc):
    print(f'testing...')
    acc_sum, loss_sum, n ,batch_count = 0.0, 0.0, 0, 0
    net.eval() #进入测试模式
    for batch in data_iter:
        seq1 = batch.seq_1[0]
        seq2 = batch.seq_2[0]
        mask1 = (seq1 == 1)
        mask2 = (seq2 == 1)
        y = batch.label
        y = y.to(device)
        y_hat = net(seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device))
        acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_count += 1
        loss_sum += loss_fc(y_hat, y).cpu().item()
    net.train() # 更改为训练模式
    return acc_sum / n, loss_sum/ batch_count

    # x.size: [batch,]
def train(net, device, train_iter, val_iter, test_iter, num_epochs, loss_fc, optimizer):
    net.train()
    print(f'training on : {device}')
    min_num =  0.6
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum,  n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for batch in train_iter:
            seq1 = batch.seq_1[0]
            seq2 = batch.seq_2[0]
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            y = batch.label
            y = y.to(device)
            y_hat = net(seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device))
            l = loss_fc(y_hat, y)
            optimizer.zero_grad() 
            l.backward()
            optimizer.step()
            train_l_sum += l
            batch_count += 1
            n += y.shape[0]
            train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()
            if batch_count % 1000 == 0:
                test_acc, test_num = val_test(net, test_iter, loss_fc)
                val_acc, val_num = val_test(net, val_iter, loss_fc)
                if test_acc >= min_num:
                    min_num = test_acc
                    print('Save model...')
                    tc.save(net.state_dict() ,'./Model/model_normal.pth')
                print('epoch: %d, train_loss: %.3f train_acc: %.3f val_loss: %.3f val_acc: %.5f test_loss: %.3f test_acc: %.5f Time: %.3F'%(epoch, train_l_sum / batch_count, train_acc_sum /n, val_num, val_acc, test_num, test_acc, time.time()-start))
                start = time.time()


# In[10]:


# 定义网络
net = ESIM_model(sequence.vocab)
# 定义设备
device = tc.device('cuda:0')
# 初始化参数
net.embedding.weight.data.copy_(sequence.vocab.vectors)
net = nn.DataParallel(net, device_ids = [0,1,2]) 
net.to(device)
# 定义损失函数
loss_fc = nn.CrossEntropyLoss()
# 定义激活函数
optimizer = tc.optim.Adam(net.parameters(),lr = args.lr)


# In[11]:


train(net, device, train_iter, val_iter, test_iter, args.num_epochs, loss_fc, optimizer)


# ## 测试

# In[35]:


# 定义设备
device = tc.device('cuda:0')
# 定义网络
net = ESIM_model(sequence.vocab)
net = nn.DataParallel(net, device_ids = [0,1,2]) 
net.load_state_dict(tc.load('./Model/model_normal.pth'))
net.to(device)
idx_to_label = ['entailment', 'neutral', 'contradiction']
def test(net, data_iter, ):
    net.eval() #进入测试模式
    rel_label = []
    for batch in data_iter:
        seq1 = batch.seq1[0]
        seq2 = batch.seq2[0]
        mask1 = (seq1 == 1)
        mask2 = (seq2 == 1)
        y_hat = net(seq1.to(device), seq2.to(device), mask1.to(device), mask2.to(device))
        y_hat = y_hat.argmax(dim=1).cpu()
        rel_label += y_hat.tolist()
    net.train() # 更改为训练模式
    with open('./test_rel.txt', 'w')  as fp:
        for label in rel_label:
            label =  idx_to_label[label]
            fp.writelines( str(label) + '\n')
    


# In[36]:


test_path = './snli.test'
examples = []
fields = [('seq1',sequence), ('seq2', sequence)]
with open(test_path, 'r') as fp:
    contents = fp.readlines()
    for content in contents:
        seqs = content.strip().split('|||')
        examples.append(data.Example.fromlist(seqs, fields))
test_dataset = data.Dataset(examples, fields)
print(len(test_dataset))
test_iter = data.Iterator(test_dataset, batch_size= args.batch_size, shuffle = False)


# In[37]:


test(net, test_iter)

