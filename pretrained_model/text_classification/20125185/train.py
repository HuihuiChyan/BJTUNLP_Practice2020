#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel, AdamW
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import time
import data_process
from model import Bert
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--max_len', type=int, default=256)
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--model_path', type=str, default='./model/best.bin1')
args = parser.parse_args(args=[])

    
train_iter, test_iter = data_process.load_data(args) 


# 定义模型、优化器、损失函数
net = Bert()
net = net.to(device)
crition = nn.CrossEntropyLoss()
optimizer = AdamW(net.parameters(),lr = args.lr)


#验证集的准确率
def test_accurate(net, data_iter, crition):
    acc_sum, loss_sum, n ,batch_num = 0.0, 0.0, 0, 0
    net.eval()
    for x,y,z in data_iter:
        #mask = (x>0).to(device)
        mask = z.to(device)
        x, y = x.to(device), y.to(device)
        y_hat = net(x, mask)
        acc_sum += (y_hat.argmax(dim=1)==y).sum().cpu().item()
        n += y.shape[0]
        batch_num += 1
        loss_sum += crition(y_hat, y).cpu().item()
    net.train()
    return acc_sum / n, loss_sum/ batch_num

#模型训练
net.train()
print(f'training on : {device}')
best_acc = 0.5
for epoch in range(args.num_epochs):
    start = time.time()
    train_l_sum,train_acc_sum,n,batch_num = 0.0, 0.0, 0, 0
    for x,y,z in train_iter:
        #mask = (x>0).to(device)
        mask = z.to(device)
        x, y = x.to(device), y.to(device)
        y_hat = net(x, mask)
        l = crition(y_hat, y)
        optimizer.zero_grad() 
        l.backward()
        optimizer.step()
        train_l_sum += l
        batch_num += 1
        n += y.shape[0]
        train_acc_sum += (y_hat.argmax(dim = 1) == y).sum().cpu().item()

    test_acc, test_loss = test_accurate(net, test_iter, crition)
    if test_acc >= best_acc:
        best_acc = test_acc
        torch.save(net.state_dict() ,args.model_path)
    print('epoch: %d train_loss: %.3f train_acc: %.4f test_loss: %.3f test_acc: %.4f time: %.3f'%(epoch+1, train_l_sum / batch_num, train_acc_sum /n, test_loss, test_acc, time.time()-start))




# # 测试结果


# def get_test_iter(path):
#     with open(path,'r') as fr:
#         lines = fr.readlines()
#     # 分词
#     tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/bert-base-uncased-vocab.txt',do_lower_case=True)
#     texts = [tokenizer.tokenize('[CLS]'+line.strip().lower()+'[SEP]') for line in lines]   
#     texts = [tokenizer.convert_tokens_to_ids(text) for text in texts]
#     texts = [data_process.padding(text,args.max_len) for text in texts]
#     mask = [[float(idx) > 0 for idx in feature] for feature in texts]
#     # 构造数据集
#     test_dataset = TensorDataset(torch.tensor(texts),torch.tensor(mask))                    
#     #构造迭代器
#     test_iter = DataLoader(test_dataset, batch_size=args.batch_size,shuffle=False)
#     return test_iter
            
# test_iter1 = get_test_iter('./data/test.txt')


# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

# model = BertModel.from_pretrained('./bert-base-uncased')
# # 定义模型、优化器、损失函数
# test_model = Bert()
# test_model = test_model.to(device)
# test_model.load_state_dict(torch.load('./model/best.bin1'))
# y_pre = []
# for x,z in test_iter1:
#     #mask = (x>0).to(device)
#     mask = z.to(device)
#     x = x.to(device)
#     y_hat = test_model(x, mask)
#     y_hat = y_hat.cpu().argmax(dim=1).tolist()
#     y_pre += y_hat
# #写入结果
# with open('./data/result.txt', 'w', encoding='utf8') as f:
#     for label in y_pre:
#         if label == 1:
#             f.write('pos'+'\n')
#         else:
#             f.write('neg'+'\n')

