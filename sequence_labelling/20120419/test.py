#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as tc
from torch import nn
import pandas as pd
from torchtext import data
import torchtext
import time
import argparse
from torch import autograd
from torch.autograd import Variable
from tkinter import _flatten
import numpy as np
import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score
from model import BiLSTM_CRF
from process import *

parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float, default = 0.01, help='学习率')
parser.add_argument('--save_path',type=str, default='./Model_1.pth',help='模型保存位置')
parser.add_argument('--char_lstm_embed_size',type=int, default= 25 , help='字符集lstm嵌入dim')
parser.add_argument('--char_lstm_hidden_size',type=int, default= 25 , help='字符集sltm隐藏层dim')
parser.add_argument('--word_embed_size',type=int, default = 200, help='word嵌入dim')
parser.add_argument('--input_embed_size',type=int, default = 250, help='lstm_input_嵌入dim')
parser.add_argument('--hidden_size',type=int , default = 250, help='decoder_lstm隐藏层dim')
parser.add_argument('--add_dropout',type= int , default = 1, help='input_embed是否dropout')
parser.add_argument('--device',type=str , default ='cuda:2', help='train device')
args = parser.parse_args(args=[])

idx_to_tag = ['B-ORG','O','B-MISC','B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'STOP', 'START']
# 获取数据迭代器
seq, char_, train_iter, test_iter, val_iter = get_data_iter()
START ='START'
STOP = 'STOP'
device = tc.device('cuda:2')
net = BiLSTM_CRF(tag_to_idx, seq.vocab, char_.vocab, args)
net.load_state_dict(tc.load(args.save_path))
net = net.to(device)

#测试
def test_(net, data_iter, device, idx_to_tag):
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    seq_pred = []
    net.eval() # 进行测试模式
    for batch_data in data_iter:  
        sentence = (batch_data.Seq).to(device)
        char_ = (batch_data.Char_).to(device)
        char_len = (batch_data.Char_len).to(device)
        tag_seq = net(sentence, char_, char_len)
        seq_pred.append(tag_seq)
        n += sentence.shape[1]
        if n % 200 == 0:
            print(f'test__ n = {n}')
    net.train()  # 进入训练模式
    seq_pred = [[idx_to_tag[idx] for idx in seq_idx]for seq_idx in seq_pred]
    return seq_pred

# In[2]:


examples = []
char_to_idx = char_.vocab.stoi
def tag_tokenizer(x):
    rel = [int(tag) for tag in x.split()]    
word_list = data.Field(sequential= True , use_vocab= False, tokenize= tag_tokenizer)
char = data.Field(sequential=True, use_vocab = False, batch_first= True)
char_len = data.Field(sequential=True, use_vocab=False,batch_first=True)
fileds = [('Seq',word_list),('Char_',char),('Char_len',char_len)]
with open('conll03.test', 'r') as fp:
    dataset = fp.readlines()
    print(len(dataset))
    for sequen in dataset:
        word_list = [seq.vocab.stoi[word.lower()] for word in sequen.strip().split()]
        char\_list = [[char_to_idx[c] for c in word] for word in sequen.strip().split()]
        char_len_list = [len(word) for word in sequen.strip().split()]
        examples.append(data.Example.fromlist([word_list, pad_char_list(char_list), char_len_list],fileds))
test_dataset = data.Dataset(examples, fileds)
test_iter = data.BucketIterator(test_dataset, batch_size=1, shuffle=False,repeat=False,sort=False,device=tc.device('cpu'))


# In[2]:


seq_pred  = test_(net, test_iter, device, idx_to_tag)


# In[4]:


with open('conll03_test_rel_2.txt', 'w', encoding = 'utf-8') as fp:
    for seq_tag in seq_pred:
        tag_list = ' '.join(seq_tag) + '\n'
        fp.writelines(tag_list)

