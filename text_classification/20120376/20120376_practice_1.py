#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import json
import datetime
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.test.utils import datapath, get_tmpfile
import torch.nn.functional as F
import os


# In[2]:


# ======================== 数据预处理 ========================

def data_preprocessing(line):
    line = line.replace('<br /><br />','')
    line = line + ' '
    change_word = ['.', '!', ',' , ':', '?', '(', ')', '/']
    for word in change_word:
        line = line.replace(word, ' '+word+' ')
    line = line.replace('  ',' ')
    line = line + '\n'
    return line

f=open('GloVe-master/corpus.txt','w')

#获得训练集文件夹
filedir_pos = 'aclImdb/train/pos/'
filedir_neg = 'aclImdb/train/neg/'

#获取当前文件夹中的文件名称列表
filenames_pos = os.listdir(filedir_pos)
filenames_neg = os.listdir(filedir_neg)
train_list, train_labels = [], []
for filename in filenames_pos:
    filepath = filedir_pos + filename
    for line in open(filepath):
        line = data_preprocessing(line)
        f.write(line)
        train_list.append(line)
        train_labels.append([1])
for filename in filenames_neg:
    filepath = filedir_neg + filename
    for line in open(filepath):
        line = data_preprocessing(line)
        f.write(line)
        train_list.append(line)
        train_labels.append([0])
f.close()

#获得验证集文件夹
filedir_pos = 'aclImdb/test/pos/'
filedir_neg = 'aclImdb/test/neg/'

#获取当前文件夹中的文件名称列表
filenames_pos = os.listdir(filedir_pos)
filenames_neg = os.listdir(filedir_neg)
test_list, test_labels = [], []
for filename in filenames_pos:
    filepath = filedir_pos + filename
    for line in open(filepath):
        test_list.append(data_preprocessing(line))
        test_labels.append([1])   
for filename in filenames_neg:
    filepath = filedir_neg + filename
    for line in open(filepath):
        test_list.append(data_preprocessing(line))
        test_labels.append([0])


# In[3]:


# ======================== 将GloVe处理成Word2Vec ========================

# glove_file = './GloVe-master/vectors.txt'
# tmp_file = './GloVe-master/word2vec_model.txt'
# _ = glove2word2vec(glove_file, tmp_file)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./GloVe-master/word2vec_model.txt',binary=False, encoding='utf-8')
vocab_list = list(w2v_model.vocab.keys())
#print(len(vocab_list)) #37125
word_index = {word: index for index, word in enumerate(vocab_list)}  #获得字典：{'the': 0, 'a': 1...}


# In[4]:


def get_dataset(dictionary, sample_list, sample_labels):
    sample_features = []                                                
    for sentence in sample_list:
        words = []
        sentence = sentence.split(" ")
        for word in sentence:
            if word not in dictionary:
                words.append(0)
            else:
                words.append(dictionary[word])
        sample_features.append(words)                                   

    sample_data = []                                                    
    for i in range(len(sample_features)):
        temp = []
        temp.append(torch.Tensor(sample_features[i]).long())
        temp.append(torch.Tensor(sample_labels[i]).long())
        sample_data.append(temp)
    
    return sample_data

train_data = get_dataset(word_index, train_list, train_labels)
test_data = get_dataset(word_index, test_list, test_labels)


# In[5]:


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
    sample_features = rnn_utils.pad_sequence(sample_features, batch_first=True, padding_value=0) 
    return sample_features, sample_labels, data_length

def test_evaluate(model, test_dataloader, batch_size, num_epoch):
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


# In[6]:


device = 'cuda:1'
loss_func = torch.nn.CrossEntropyLoss()

# 让Embedding层使用训练好的Word2Vec权重
embedding_matrix = w2v_model.vectors
input_size = embedding_matrix.shape[0]   #37125, 词典的大小
hidden_size = embedding_matrix.shape[1]  #50, 隐藏层单元个数
kernel_size = [3, 4, 5]
nums_channels = [100, 100, 100]
model = TextCNN(input_size, hidden_size, 2, embedding_matrix, kernel_size, nums_channels).to(device)
model.load_state_dict(torch.load('./model_save/TextCNN_save_2.pt'))
print("load model...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.0000001, weight_decay=0.1)#

batch_size = 256
num_epoch = 50

train_dataloader = DataLoader(train_data, batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size, collate_fn=collate_fn, shuffle=True)

train_loss = []
train_accuracy = []
test_loss = []
test_accuracy = []
loss_min = 0.302007
for epoch in range(num_epoch):
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
    test_l, test_a = test_evaluate(model, test_dataloader, batch_size, num_epoch)
    test_loss.append(test_l)
    test_accuracy.append(test_a)
    end = datetime.datetime.now()
    print('epoch %d, train_loss %f, train_accuracy %f, test_loss %f, test_accuracy %f, time %s'% 
          (epoch+1, train_loss[epoch], train_accuracy[epoch], test_loss[epoch], test_accuracy[epoch], end-start))
    if test_loss[epoch] < loss_min:
        loss_min = test_loss[epoch]
        torch.save(model.state_dict(), './model_save/TextCNN_save_2.pt')
        print("save model...")


# In[ ]:




