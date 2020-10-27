#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch as tc
from transformers import AdamW, BertTokenizer, BertModel, BertForMaskedLM, AutoModelForMaskedLM, get_linear_schedule_with_warmup
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import argparse
import time


# In[14]:


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size',type=int, default=16, help = '样本批次')
parser.add_argument('--max_len',type=int, default=256, help = 'bert预训练模型最大seq_len')
parser.add_argument('--num_layers',type=int, default=2, help = '分类类别个数')
parser.add_argument('--lr',type=float, default=2e-5, help = '学习率')
parser.add_argument('--num_epochs',type=int, default=10, help = '训练批次')
parser.add_argument('--max_grad_norm',type=float, default=1.0, help = '梯度裁剪')
parser.add_argument('--warmup_proportion',type=float, default=0.1, help = '上设截至位置占所有步长的比例')
args = parser.parse_args()


# In[15]:


tokenizer = BertTokenizer.from_pretrained('./Vocab/bert-base-uncased-vocab.txt', do_lower_case=True) # 加载词典
model = BertModel.from_pretrained('./Model_english')  # 加载预训练bert模型
config = model.config


# In[11]:


# 处理数据
def get_dataiter(csv_path, tokenizer, max_len, batch_size, train = False):
    def process_seq(content):
        #获取句子的前128个单词和后128个单词 
        if len(content) > args.max_len:
            return content[:128] + content[-128:]
        return content + [0] * (max_len-len(content)) 
    def pad(content):
        return content[:max_len] if len(content) > max_len else content + [0]*(max_len-len(content)) 
    pro_data = pd.read_csv(csv_path)
    data_set = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize('[CLS]' + content.lower() + '[SEP]')) for content in pro_data.content ]
    features = [process_seq(content_idx) for content_idx in data_set]
    label = pro_data.label
    mask = [[float(idx) > 0 for idx in feature] for feature in features]
    dataset = TensorDataset(tc.tensor(features), tc.tensor(label), tc.tensor(mask))
    print(len(dataset))
    if train:
        dataiter = DataLoader(dataset, batch_size = batch_size, shuffle = True)
    else:
        dataiter = DataLoader(dataset, batch_size = batch_size)
    return dataiter

train_path = './Dataset/train.csv'
test_path = './Dataset/test.csv'
train_iter = get_dataiter(train_path, tokenizer, args.max_len, args.batch_size, train = True)
test_iter = get_dataiter(test_path, tokenizer, args.max_len, args.batch_size)


# In[16]:


# 定义模型
class bert_for_textclassification(nn.Module):
    def __init__(self, args, config, bert_model):
        super(bert_for_textclassification, self).__init__()
        self.bert_model =bert_model
        self.classifier = nn.Linear(config.hidden_size, args.num_layers)
        self.dropout = nn.Dropout(0.5)
    def forward(self, features, mask, labels):
        output = self.bert_model(features, attention_mask = mask)
        pooled_output  = output[1]
        pooled_output = self.dropout(pooled_output)
        y_hat = self.classifier(pooled_output)
        return y_hat


# In[23]:


#验证集的准确率
def val_test(net, data_iter, loss_fc):
    acc_sum, loss_sum, n ,batch_count = 0.0, 0.0, 0, 0
    net.eval() # 进行评估模式
    for batch in data_iter:
        batch = tuple(t.to(device) for t in batch)
        features, labels, masks = batch
        y_hat = net(features, masks, labels)
        acc_sum += (y_hat.argmax(dim=1)==labels).sum().cpu().item()
        n += labels.shape[0]
        batch_count += 1
        loss_sum += loss_fc(y_hat, labels).cpu().item()
    net.train() # 更改为训练模式
    return acc_sum / n, loss_sum/ batch_count

def train(net, device, train_iter, val_iter, num_epochs, loss_fc, optimizer):
    print(f'training on : {device}')
    min_num = 0.8
    net.train()  #因为预训练模型是eval模式的，所以需要先进入train模式
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum,  n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        for batch in train_iter:
            batch = tuple(t.to(device) for t in batch)
            features, labels, masks = batch
            y_hat = net(features, masks, labels)
            l = loss_fc(y_hat, labels)
            optimizer.zero_grad() 
            l.backward()
            optimizer.step()
            train_l_sum += l
            batch_count += 1
            n += labels.shape[0]
            train_acc_sum += (y_hat.argmax(dim = 1) == labels).sum().cpu().item()
            if batch_count % 100 == 0:
                print(f'epoch : {epoch} train_step :{batch_count}---')
        val_acc, val_num = val_test(net, val_iter, loss_fc)
        if val_acc >= min_num:
            min_num = val_acc
            print('Save model...')
            tc.save(net.state_dict() ,'./Model/model_normal02.pth')
        print('epoch: %d, train_loss: %.3f train_acc: %.3f test_loss: %.3f val_acc: %.3f Time: %.3F'%(epoch, train_l_sum / batch_count, train_acc_sum /n, val_num, val_acc, time.time()-start))


# In[24]:


# 获取数据迭代器
device = tc.device('cuda:2')
# 定义模型
net = bert_for_textclassification(args, config, model)
net = nn.DataParallel(net, device_ids=[2,0,1])
net = net.to(device)
# 交叉熵损失函数
loss_fc = nn.CrossEntropyLoss() 
#定义激活函数
param_optimizer = list(net.named_parameters())
no_decay = ['bias', 'gamma', 'beta']   #这三个变量不做梯度微调
optimizer_grouped_parameters = [
    {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
    'weight_decay_rate':0.01},
    {'params':[p for n, p in param_optimizer if  any(nd in n for nd in no_decay)],
    'weight_decay_rate':0.0},
]
optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
# num_training_steps = int(
#             25000 / args.batch_size * args.num_epochs)
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_training_steps * args.warmup_proportion, num_training_steps=num_training_steps) 


# In[25]:


train(net, device, train_iter, test_iter, args.num_epochs, loss_fc, optimizer)

