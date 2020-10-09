#!/usr/bin/env python
# coding: utf-8


import gensim
import json
import datetime
import numpy as np
import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from gensim.models import Word2Vec
from collections import Counter
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from sacremoses import MosesTokenizer

# ======================== 数据预处理 ========================


f_train_pos = open('train_pos.txt', 'w')
f_train_neg = open('train_neg.txt', 'w')
f_valid_pos = open('valid_pos.txt', 'w')
f_valid_neg = open('valid_neg.txt', 'w')

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
        f_train_pos.write(line+'\n')
for filename in filenames_neg:
    filepath = filedir_neg + filename
    for line in open(filepath):
        f_train_neg.write(line+'\n')

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
        f_valid_pos.write(line+'\n')
for filename in filenames_neg:
    filepath = filedir_neg + filename
    for line in open(filepath):
        f_valid_neg.write(line+'\n')

f_train_neg.close()
f_train_pos.close()
f_valid_neg.close()
f_valid_pos.close()