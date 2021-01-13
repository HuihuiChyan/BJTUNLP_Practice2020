#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel,AdamW
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import time


# 获取文件名列表
def get_file_list(path):
    file_list = []
    for file_name in os.listdir(path):
        file_list.append(path + file_name)
    return file_list

# 从文件中读取数据
def read_datas(pos,neg):
    texts, labels = [],[]
    # pos类别---1
    for file_name in pos: 
        with open(file_name,'r') as fr:
            texts.append('[CLS]'+fr.readline().strip().lower()+'[SEP]')
            labels.append(1)
    # neg类别---0
    for file_name in neg:
        with open(file_name,'r') as fr: 
            texts.append('[CLS]'+fr.readline().strip().lower()+'[SEP]')
            labels.append(0)
    return texts,labels

def padding(text,max_len):
    if len(text) < max_len:
        text = text + [0]*(max_len-len(text))
    else:
        text = text[:max_len-1] +[text[-1]] 
    return text  
    

def load_data(args):
    train_pos = get_file_list('./data/aclImdb/train/pos/')
    train_neg = get_file_list('./data/aclImdb/train/neg/')
    test_pos = get_file_list('./data/aclImdb/test/pos/')
    test_neg = get_file_list('./data/aclImdb/test/neg/')
    train_texts,train_labels = read_datas(train_pos,train_neg)
    test_texts,test_labels = read_datas(test_pos,test_neg)
    
    # 分词
    tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt',do_lower_case=True)
    train_texts = [tokenizer.tokenize(text) for text in train_texts]
    train_features = [tokenizer.convert_tokens_to_ids(text) for text in train_texts]
    test_texts = [tokenizer.tokenize(text) for text in test_texts]
    test_features = [tokenizer.convert_tokens_to_ids(text) for text in test_texts]
    # 
    train_features = [padding(text,args.max_len) for text in train_features]
    test_features = [padding(text,args.max_len) for text in test_features]
    
    train_mask = [[float(idx) > 0 for idx in feature] for feature in train_features]
    test_mask = [[float(idx) > 0 for idx in feature] for feature in test_features]
    
    # 构造数据集
    train_dataset = TensorDataset(torch.tensor(train_features),torch.tensor(train_labels),torch.tensor(train_mask))
    test_dataset = TensorDataset(torch.tensor(test_features),torch.tensor(test_labels),torch.tensor(test_mask))
                        
    #构造迭代器
    train_iter = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    test_iter = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
    
    return train_iter, test_iter
