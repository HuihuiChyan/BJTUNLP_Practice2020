#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import re
import os
import random
import time
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torchtext.vocab as Vocab
import torch.utils.data as Data
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict,Counter
from sacremoses import MosesTokenizer
import gensim
# glove预训练词向量
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
# glove-->word2vec
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.metrics import accuracy_score


# In[2]:


def read_data(data_root,folder):
    data = []
    for label in ['pos', 'neg']:
        folder_name = data_root+folder+label+"/"
        for file in tqdm(os.listdir(folder_name)):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


# In[3]:


data_root = "data/aclImdb/"
train_data, test_data = read_data(data_root,'train/'), read_data(data_root,'test/')

for sample in train_data[:5]:
    print(sample[0][:50], '\t', sample[1])


# In[4]:


train_labels = list(np.array(train_data).T[1])
test_labels = list(np.array(test_data).T[1])
train_label=[int(each) for each in train_labels]
test_label=[int(each) for each in test_labels]


# In[5]:


check_data = []
with open('data/test.txt', 'r', encoding='utf-8') as fr:
    for line in fr.readlines():
        check_data.append([line.strip().replace('\n','').lower()])


# In[6]:


def get_tokenized(data):
    text=[]
    for i in range(len(data)):
        str = re.sub('[^\w ]','',data[i][0])
        text.append(nltk.word_tokenize(str))
    return text


# In[7]:


def get_tokenized_check(data):
    text = []
    for i in range(len(data)):
        s = ''.join(data[i])
        str = re.sub('[^\w ]','',s)
        text.append(nltk.word_tokenize(str))
    return text


# In[8]:


train_text = get_tokenized(train_data)
test_text = get_tokenized(test_data)
check_text = get_tokenized_check(check_data)


# In[9]:


if os.path.exists('data/vocab.txt'):
    with open('data/vocab.txt','r',encoding='utf-8') as fr:
        vocab_words = [line.strip() for line in fr.readlines()]
else:
    train_words=[]
    for line in train_text:
        train_words.extend(line)
    counter=Counter(train_words)
    common_words=counter.most_common()
    vocab_words = ['[unk]','[pad]']+[word[0] for word in common_words]
    # 词表
    fw = open('data/vocab.txt','w',encoding='utf-8')
    for word in vocab_words:
        fw.write(word+'\n')


# In[10]:


# word转换为idx
word_to_idx = defaultdict(lambda :0)  #默认第0个词UNK
idx_to_word = defaultdict(lambda :'[unk]')
for idx, word in enumerate(vocab_words):
    word_to_idx[word] = idx
    idx_to_word[idx] = word


# In[12]:


# 最大300 不足补齐
train_text = [line[:300] for line in train_text]
test_text = [line[:300] for line in test_text]
train_text = [line + ['[pad]' for i in range(300-len(line))] for line in train_text]
test_text = [line + ['[pad]' for i in range(300-len(line))] for line in test_text]
check_text = [line[:300] for line in check_text]
check_text = [line + ['[pad]' for i in range(300-len(line))] for line in check_text]


# In[13]:


train_text_idx = [[word_to_idx[word] for word in text] for text in train_text]
test_text_idx = [[word_to_idx[word] for word in text] for text in test_text]
check_text_idx = [[word_to_idx[word] for word in text] for text in check_text]


# In[14]:


train_text_idx = torch.tensor(train_text_idx)
test_text_idx = torch.tensor(test_text_idx)
train_label = torch.tensor(train_label)
test_label = torch.tensor(test_label)


# In[15]:


check_text_idx = torch.tensor(check_text_idx)


# In[16]:


train_dataset = torch.utils.data.TensorDataset(train_text_idx, train_label)
test_dataset = torch.utils.data.TensorDataset(test_text_idx, test_label)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128,shuffle=True)

check_dataset = torch.utils.data.TensorDataset(check_text_idx, train_label)
check_loader = torch.utils.data.DataLoader(dataset=check_dataset, batch_size=128,shuffle=False)


# In[17]:


if os.path.exists('data/word2vec.txt'):
    wvmodel = KeyedVectors.load_word2vec_format('data/word2vec.txt', binary=False, encoding='utf-8')
else:
    path = os.getcwd()
    glove_file = datapath(os.path.join(path, 'data/glove.6B.300d.txt'))
    tmp_file = get_tmpfile(os.path.join(path, "data/word2vec.txt"))
    glove2word2vec(glove_file, tmp_file)
    wvmodel = KeyedVectors.load_word2vec_format(tmp_file, binary=False, encoding='utf-8')


# In[31]:


class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, droput, **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.embedding_S = nn.Embedding(vocab_size, embed_size)
        self.embedding_S.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_S.weight.requires_grad = False
        self.embedding_D = nn.Embedding(vocab_size, embed_size)
        self.embedding_D.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_D.weight.requires_grad = True
        num_filters = 256
        self.labels = labels
        self.conv1 = nn.Conv2d(1,num_filters,(2,embed_size))
        self.conv2 = nn.Conv2d(1,num_filters,(3,embed_size))
        self.conv3 = nn.Conv2d(1,num_filters,(4,embed_size))
        self.conv4 = nn.Conv2d(1,num_filters,(5,embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 2 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool4 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.dropout = nn.Dropout(droput)
        self.linear = nn.Linear(2*num_filters*4,labels)
    def forward(self, x):
        #两层通道
        out1 = self.embedding_S(x).view(x.shape[0],1,x.shape[1],-1)
        out2 = self.embedding_D(x).view(x.shape[0],1,x.shape[1],-1)
        #第一层卷积
        x1 = F.relu(self.conv1(out1))
        x2 = F.relu(self.conv2(out1))
        x3 = F.relu(self.conv3(out1))
        x4 = F.relu(self.conv4(out1))
        #第一层池化
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        #第二层卷积
        x5 = F.relu(self.conv1(out2))
        x6 = F.relu(self.conv2(out2))
        x7 = F.relu(self.conv3(out2))
        x8 = F.relu(self.conv4(out2))
        #第二层池化
        x5 = self.pool1(x5)
        x6 = self.pool2(x6)
        x7 = self.pool3(x7)
        x8 = self.pool4(x8)

        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        x = x.view(x.shape[0], 1, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = x.view(-1, self.labels)
        return x



# In[32]:


embed_size = 300
labels = 2
lr = 0.0001
use_gpu = True
seq_len = 300
droput = 0.5
num_epochs = 50
best_acc = 0


# In[37]:


word2vec_path = "data/word2vec.txt"
output_path = "res22.txt"


# In[34]:


weight = torch.zeros(len(vocab_words)+1, embed_size)


# In[35]:


for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        print("wrong!")
        continue
    if(idx_to_word[word_to_idx[wvmodel.index2word[i]]] == '[unk]' or '[pad]'):
        continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
print("weight dealt over~~~~")


# In[36]:


net = textCNN(vocab_size=len(vocab_words)+1, embed_size=embed_size, seq_len=seq_len, labels=labels, weight=weight,
              droput=droput)
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
net = nn.DataParallel(net)
net = net.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)


# In[ ]:


print("start training!")
for epoch in range(num_epochs):
    f = open(output_path, "a")
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    net.train()
    for feature, label in train_loader:
        n += 1
        feature = feature.cuda()
        label = label.cuda()
        optimizer.zero_grad()
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
        train_loss += loss
    with torch.no_grad():
        net.eval()
        for test_feature, test_label in test_loader:
            m += 1
            test_feature = test_feature.cuda()
            test_label = test_label.cuda()
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data, dim=1), test_label.cpu())
            test_losses += test_loss
    end = time.time()
    runtime = end - start
    f = open(output_path, "a")
    f.write(
        'epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best acc: %.5f,time: %.4f \n' % (
            epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_acc / m,
            runtime))
    f.close()

    #kore delete previous
    if (test_acc > best_acc and test_acc / m > 0.85):
        best_acc = test_acc
        torch.save(net, 'best22.pth')
        print("test_acc / m > 0.85")
        f = open("pre22.txt", "a")
        net.eval()
        for x, y in check_loader:
            x = x.cuda()
            test_score = net(x)
            print(torch.argmax(test_score.cpu().data, dim=1))
            f.write(str(torch.argmax(test_score.cpu().data, dim=1)))
        f.close()
        break

