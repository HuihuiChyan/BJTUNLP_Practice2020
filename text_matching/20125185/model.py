#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import numpy as np
import pdb
import csv
from torchtext import data
from torchtext.vocab import Vectors
import torch.nn.functional as F
import torch.optim as optim
import argparse
import time


class ESIM(nn.Module):
    def __init__(self,args,vocab):
        super(ESIM,self).__init__()
        self.embedding = nn.Embedding(len(vocab),args.embedding_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        self.embedding.weight.requires_grad = False
        
        self.lstm1 = nn.LSTM(args.embedding_dim, args.hidden_dim,batch_first=True,bidirectional=True)
        self.lstm2 = nn.LSTM(args.hidden_dim, args.hidden_dim,batch_first=True,bidirectional=True)
        self.linear1 = nn.Linear(args.hidden_dim * 8, args.hidden_dim)
        self.linear2 = nn.Linear(args.hidden_dim, 3)
        self.dropout = nn.Dropout(args.dropout)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.tanh = nn.Tanh()
    
    def soft_attention_align(self,x1,x2,mask1,mask2):        
        '''
        x1,x2:   batch,max_len,hidden_dim
        '''
        # attention:   batch,max_len,max_len
        attention1 = torch.matmul(x1, x2.transpose(1,2))
        attention2 = torch.matmul(x2, x1.transpose(1,2))
        # mask操作
        attention1 = attention1.masked_fill(mask2.unsqueeze(-1)==1,-1e9)
        attention2 = attention2.masked_fill(mask1.unsqueeze(-1)==1,-1e9)
        # softmax操作
        weight1 = F.softmax(attention1, dim=-1)
        weight2 = F.softmax(attention2, dim=-1)
        # 得到soft-attention对齐后的表示
        # x1_align:   batch,max_len,hidden_dim
        x1_align = torch.matmul(weight1,x2)
        x2_align = torch.matmul(weight2,x1)
        return x1_align,x2_align
    
    def sub_and_mul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub,mul],-1)
    
    def avg_and_max_pool(self, x):
        # x:  batch, max_len, (2*hidden_dim)
        p1 = F.avg_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2),x.size(1)).squeeze(-1)
        # output: batch, 4*hidden_dim
        return torch.cat([p1,p2],dim=-1)
    
    def forward(self, sent1, sent2, mask1, mask2):
        # batch,max_len------>batch, max_len, embedding_dim
        embeds1 = self.embedding(sent1)
        embeds2 = self.embedding(sent2)
        
        # local——lstm+soft-attention
        # batch, max_len, embedding_dim--->batch, max_len, hidden_dim
        o1, _ = self.lstm1(embeds1)
        o2, _ = self.lstm1(embeds2)
        sent1_align, sent2_align = self.soft_attention_align(o1,o2,mask1,mask2)
        
        # enhance——拼接+全连接
        # ---->batch, max_len, 8*hidden_dim
        m1 = torch.cat((o1,sent1_align,self.sub_and_mul(o1,sent1_align)),dim=-1)
        m2 = torch.cat((o2,sent2_align,self.sub_and_mul(o2,sent2_align)),dim=-1)
        m1_ = self.relu(self.linear1(m1))
        m2_ = self.relu(self.linear1(m2))
        
        # composition——组合,lstm+pool+全连接
        # ---->batch, max_len, 2*hidden_dim
        sent1_compose,_ = self.lstm2(m1_)
        sent2_compose,_ = self.lstm2(m2_)
        # ---->batch, 4*hidden_dim
        sent1_pool = self.avg_and_max_pool(sent1_compose)
        sent2_pool = self.avg_and_max_pool(sent2_compose)
        
        x = torch.cat([sent1_pool,sent2_pool],dim=-1)
        x = self.tanh(self.linear1(x))
        x = self.dropout(x)
        output = self.linear2(x)
        return output
        