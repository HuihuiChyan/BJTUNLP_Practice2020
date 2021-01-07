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
import data_process
from model import ESIM
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_dim', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--model_path', type=str, default='./model/best.bin1')
args = parser.parse_args()
label2idx = {'entailment': 0, 'neutral': 1, 'contradiction': 2}


train_iter,dev_iter,vocab = data_process.load_data(args,device)

# 定义模型、优化器、损失函数
net = ESIM(args, vocab)
net.to(device)
crition = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = args.lr)


#验证集的准确率
def val_test(net, data_iter, crition):
    acc_sum, loss_sum, n ,batch_num = 0.0, 0.0, 0, 0
    net.eval()
    for batch in data_iter:
        sent1,sent2 = batch.sentence1[0],batch.sentence2[0]
        mask1 = (sent1 == 1)
        mask2 = (sent2 == 1)
        y = batch.label
        y = y.to(device)
        y_hat = net(sent1.to(device), sent2.to(device), mask1.to(device), mask2.to(device))
        acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_num += 1
        loss_sum += crition(y_hat, y).cpu().item()
    net.train()
    return acc_sum / n, loss_sum/ batch_num


# 训练
net.train()
print(f'training on : {device}')
best_acc = 0.5
for epoch in range(args.num_epochs):
    start = time.time()
    train_l_sum,train_acc_sum,n,batch_num = 0.0, 0.0, 0, 0
    for batch in train_iter:
        sent1,sent2 = batch.sentence1[0],batch.sentence2[0]
        mask1 = (sent1 == 1)
        mask2 = (sent2 == 1)
        y = batch.label
        y = y.to(device)
        y_hat = net(sent1.to(device), sent2.to(device), mask1.to(device), mask2.to(device))
        l = crition(y_hat, y)
        optimizer.zero_grad() 
        l.backward()
        optimizer.step()
        train_l_sum += l
        batch_num += 1
        n += y.shape[0]
        train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()
        if batch_num % 500 == 0:
            val_acc, val_loss = val_test(net, dev_iter, crition)
            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(net.state_dict() ,args.model_path)
            print('batch: %d, train_loss: %.3f train_acc: %.4f val_loss: %.3f val_acc: %.4f time: %.3F'%(batch_num, train_l_sum / batch_num, train_acc_sum /n, val_loss, val_acc, time.time()-start))
            



# # 加载测试数据
# def load_test_data(args):
#     text = data.Field(sequential= True, use_vocab = True, lower = True, fix_length = args.max_len, include_lengths = True,batch_first = True)
#     fields = [('sentence1',text),('sentence2',text)]
#     # 读取数据
#     test = []
#     with open('./data/snli.test','r') as f:
#         for line in f.readlines():
#             sents = line.strip().split('|||')
#             test.append(data.Example.fromlist([sents[0],sents[1]],fields))
#     text.vocab = vocab
#     # 构造数据集
#     test_datasets = data.Dataset(test, fields)
#     # 构造迭代器
#     test_iter = data.Iterator(test_datasets,
#                                 shuffle=False,
#                                 batch_size=args.batch_size,
#                                 device=device)  
#     return test_iter
# test_iter = load_test_data(args)

# # 加载模型数据
# net = ESIM(args, vocab)
# net.load_state_dict(torch.load('./model/best.bin'))
# net.to(device)

# # 测试
# net.eval()
# label_idxs = []
# for batch in test_iter:
#     sent1,sent2 = batch.sentence1[0],batch.sentence2[0]
#     mask1 = (sent1 == 1)
#     mask2 = (sent2 == 1)
#     y_pre = net(sent1.to(device), sent2.to(device), mask1.to(device), mask2.to(device))
#     y_pre = y_pre.argmax(dim=1).cpu()
#     label_idxs += y_pre.tolist()

# # 写入结果
# idx2label = ['entailment', 'neutral', 'contradiction']
# with open('./data/result.txt','w') as f:
#     for idx in label_idxs:
#         label = idx2label[idx]
#         f.write(label)
#         f.write('\n')



