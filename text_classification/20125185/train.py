#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
import time
from sklearn.metrics import accuracy_score
import data_process2
from model import TextCNN
import argparse
import pickle

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")



# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=80000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_len', type=int, default=2048)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--output_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--output_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--kernal_sizes', type=list, default=[2,3,4,5])
parser.add_argument('--test_per_step', type=int, default=100)
parser.add_argument('--pretrained', type=bool, default=True)

args = parser.parse_args()


# 加载数据
train_iter,test_iter,vocab_words = data_process2.load_data(args)
args.vocab_size = len(vocab_words)
print('vocab_size:',len(vocab_words))

# 加载预训练词向量
with open('./data/wvmodel.pkl', 'rb') as inp:
    wvmodel = pickle.load(inp)
print('wvmodel loaded!')

weight = torch.zeros(args.vocab_size, args.embedding_size)
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index,:] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))

# 加载模型
net = TextCNN(args,weight).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 使用交叉熵损失函数
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, net.parameters()),lr = args.lr)

# 模型训练
print('training on ',device)
best_acc = 0.0
step = 0
for epoch in range(1, args.num_epochs + 1):
    net.train()
    for X,y in train_iter:
        X, y = X.to(device), y.to(device)
        y_hat = net(X) # 计算预测概率值
        loss = criterion(y_hat, y) # 计算loss值
        optimizer.zero_grad()  # 梯度置零
        loss.backward()   # 反向传播
        optimizer.step()  # 参数更新
        step+=1

        # 测试
        if step % args.test_per_step == 0:    
            net.eval()
            all_pre=[]
            all_label=[]

            for X,y in test_iter:
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                y_pre = torch.argmax(y_hat, dim=-1)
                all_pre.extend(y_pre.tolist())
                all_label.extend(y.tolist())
            test_acc_sum = sum([int(line[0]==line[1]) for line in zip(all_pre,all_label)])
            test_acc = test_acc_sum/len(all_label)

            print('train_step %d, loss: %.4f, test_acc: %.4f'%(step, loss, test_acc))

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(net.state_dict(),'./model/best_model.bin')
print('best_acc: ', best_acc)



