##!/usr/bin/env python
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


# 获取数据（需要的是0、5、6列）
def load_data(args,device):
    text = data.Field(sequential= True, use_vocab = True, lower = True, fix_length = args.max_len, include_lengths = True,batch_first = True)
    label = data.Field(sequential = False, use_vocab = False)
    fields = [('sentence1',text),('sentence2',text),('label',label)]
    # 读取数据
    train = []
    with open('./data/snli_1.0_train.txt','r') as f:
        for line in f.readlines():
            sents = line.strip().split('\t')
            if sents[0]=='gold_label' or sents[0]=='-':
                continue
            train.append(data.Example.fromlist([sents[5],sents[6],label2idx[sents[0]]],fields))
    dev = []
    with open('./data/snli_1.0_dev.txt','r') as f:
        for line in f.readlines():
            sents = line.strip().split('\t')
            if sents[0]=='gold_label' or sents[0]=='-':
                continue
            dev.append(data.Example.fromlist([sents[5],sents[6],label2idx[sents[0]]],fields))
    
    # 构造数据集
    train = train
    dev = dev
    train_datasets = data.Dataset(train, fields)
    dev_datasets = data.Dataset(dev, fields)
    
    # 构建词表
    text.build_vocab(train_datasets,vectors = Vectors(name='./data/glove.840B.300d.txt'))
    
    # 构造迭代器
    train_iter = data.BucketIterator(train_datasets,
                                shuffle=True,
                                batch_size=args.batch_size,
                                device=device)
    
    dev_iter = data.BucketIterator(dev_datasets,
                                shuffle=False,
                                batch_size=args.batch_size,
                                device=device)
    
    return train_iter,dev_iter,text.vocab

