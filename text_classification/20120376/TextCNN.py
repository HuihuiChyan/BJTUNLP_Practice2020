#!/usr/bin/env python
# coding: utf-8

import gensim
import datetime
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import Word2Vec
from collections import Counter
import argparse
from sklearn.metrics import accuracy_score
import os

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--ckp', type=str, default='./ckp/model_2.pt')
parser.add_argument('--acc_min', type=float, default=0.911468)
parser.add_argument('--nums_channels', type=int, default=50)
args = parser.parse_args()#args=[])

acc_min = args.acc_min
if torch.cuda.is_available():
    print("using cuda")
    device = torch.device('cuda:1')

# ======================== 数据预处理 ========================

#读入训练集
with open('./train.txt', encoding='utf-8') as ftrain_feature:
    train_feature_line = [line.strip() for line in ftrain_feature.readlines()]
train_label_line = [1] * int(len(train_feature_line)/2)
temp = [0] * int(len(train_feature_line)/2)
train_label_line.extend(temp)

#读入验证集
with open('./valid.txt', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
test_label_line = [1] * int(len(test_feature_line)/2)
temp = [0] * int(len(test_feature_line)/2)
test_label_line.extend(temp)

#用split分隔开存入列表
train_feature_line = [line.split(" ") for line in train_feature_line]
test_feature_line = [line.split(" ") for line in test_feature_line]

# ======================== 将GloVe处理成Word2Vec ========================

""" glove_file = './GloVe-master/vectors.txt'
tmp_file = './GloVe-master/word2vec_model.txt'
_ = glove2word2vec(glove_file, tmp_file)"""

#获得单词字典
#model_word2vec = Word2Vec([['[UNK]'],['[PAD]']]+train_feature_line, sg=1, min_count=1, size=128, window=5)
#model_word2vec.save('word2vec_model.txt')
#w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GloVe-master/word2vec_model.txt',binary=False, encoding='utf-8')
w2v_model = Word2Vec.load('word2vec_model.txt')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词

#获得数据序列
train_feature = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature_line]
test_feature = [[word2id[word] if word in word2id else unk for word in line] for line in test_feature_line]

def get_dataset(dictionary, sample_features, sample_labels):
    sample_data = []                                                    
    for i in range(len(sample_features)):
        temp = []
        temp.append(torch.Tensor(sample_features[i]).long())
        temp.append(torch.Tensor([sample_labels[i]]).long())
        sample_data.append(temp)
    return sample_data

train_data = get_dataset(word2id, train_feature, train_label_line)
test_data = get_dataset(word2id, test_feature, test_label_line)

class TextCNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embeding_vector, kernel_sizes, num_channels):
        super().__init__()
        self.hidden_size = hidden_size
        #不参与训练的嵌入层
        self.embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  #使用预训练的词向量
        self.embedding.weight.requires_grad = False
        #参与训练的嵌入层
        self.constant_embedding = torch.nn.Embedding(num_embeddings=input_size, embedding_dim=hidden_size)
        self.constant_embedding.weight.data.copy_(torch.from_numpy(embeding_vector))  #使用预训练的词向量
        self.dropout = torch.nn.Dropout(0.5)
        self.out_linear = torch.nn.Linear(sum(num_channels), output_size)
        self.pool = GlobalMaxPool1d()
        self.convs = torch.nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(torch.nn.Conv1d(in_channels=2*hidden_size, out_channels=c, kernel_size=k))
        
    def forward(self, x):
        embeddings = torch.cat((self.embedding(x), self.constant_embedding(x)), dim=2).permute(0,2,1)
        out = torch.cat([self.pool(F.relu(conv(embeddings))).squeeze(-1) for conv in self.convs], dim=1)
        out = self.out_linear(self.dropout(out))
        return out

class GlobalMaxPool1d(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return F.max_pool1d(x, kernel_size = x.shape[2])


def collate_fn(sample_data):
    sample_data.sort(key=lambda data: len(data[0]), reverse=True)                          #倒序排序
    sample_features = []
    sample_labels = []
    for data in sample_data:
        sample_features.append(data[0])
        sample_labels.append(data[1])
    data_length = [len(data[0]) for data in sample_data]                                   #取出所有data的长度             
    sample_features = rnn_utils.pad_sequence(sample_features, batch_first=True, padding_value=padding_value) 
    return sample_features, sample_labels, data_length

def test_evaluate(model, test_dataloader, batch_size):
    test_l, test_a, n = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for data_x, data_y, _ in test_dataloader:
            label = torch.Tensor(data_y).long().to(device)
            out = model(data_x.to(device))
            prediction = out.argmax(dim=1)
            loss = loss_func(out, label)
            prediction = out.argmax(dim=1).data.cpu().numpy()
            label = label.data.cpu().numpy()
            test_l += loss.item()
            test_a += accuracy_score(label, prediction)
            n += 1
    return test_l/n, test_a/n



loss_func = torch.nn.CrossEntropyLoss()
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   #37125, 词典的大小
hidden_size = embedding_matrix.shape[1]  #50, 隐藏层单元个数
kernel_size = [3, 4, 5]
nums_channels = [args.nums_channels, args.nums_channels, args.nums_channels]
model = TextCNN(input_size, hidden_size, 2, embedding_matrix, kernel_size, nums_channels).to(device)
if os.path.exists(args.ckp):
    print("loading model......")
    model.load_state_dict(torch.load(args.ckp))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) #

train_dataloader = DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=True)

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
for epoch in range(args.num_epoch):
    model.train()
    train_l, train_a, n = 0.0, 0.0, 0
    start = datetime.datetime.now()
    for data_x, data_y, _ in train_dataloader:
        label = torch.Tensor(data_y).long().to(device)
        out = model(data_x.to(device))
        loss = loss_func(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prediction = out.argmax(dim=1).data.cpu().numpy()
        label = label.data.cpu().numpy()
        train_l += loss.item()
        train_a += accuracy_score(label, prediction)
        n += 1
    #训练集评价指标
    train_loss.append(train_l/n)
    train_accuracy.append(train_a/n)
    #测试集评价指标
    test_l, test_a = test_evaluate(model, test_dataloader, args.batch_size)
    test_loss.append(test_l)
    test_accuracy.append(test_a)
    end = datetime.datetime.now()
    print('epoch %d, train_loss %f, train_accuracy %f, test_loss %f, test_accuracy %f, time %s'% 
          (epoch+1, train_loss[epoch], train_accuracy[epoch], test_loss[epoch], test_accuracy[epoch], end-start))
    if test_accuracy[epoch] > acc_min:
        acc_min = test_accuracy[epoch]
        torch.save(model.state_dict(), args.ckp)
        print("save model...")


# In[ ]:




