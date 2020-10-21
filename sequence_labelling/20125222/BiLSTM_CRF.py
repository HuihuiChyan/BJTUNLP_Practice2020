import re
import os

import numpy as np

from itertools import chain
import os
import torch
import gensim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import accuracy_score
#返回数组，分别为文本+特征
def read_file(data_path, label_path):
	data = []
	labels = []
	for line in open(data_path):
		data.append(line.strip().split(' '))
	for line in open(label_path):
		labels.append(line.strip().split(' '))
	return data, labels
def get_stop_words_list(filepath):
	stop_words_list = []
	with open(filepath,encoding='utf8') as f:
		for line in f.readlines():
			stop_words_list.append(line.strip())
	return stop_words_list
# 特殊处理数据+去标点
def data_process(text):
    text = text.lower()
    # 特殊数据处理，该地方参考的殷同学的
    text = text.replace("<br /><br />", "").replace("'s", "is").replace("i'm", "i am").replace("he's",
                                                                                                    "he is")
    # 去除标点
    #text = re.sub("[^a-zA-Z']", "", text.lower())
    text = " ".join([word for word in text.split(' ')])
    return text
def get_token_text(text):
    token_data = [data_process(st) for st in text]
    # token_data = [st.lower() for st in text.split()]
    token_data = list(filter(None, token_data))
    return token_data
#返回文本分词形式
def get_token_data(data):
	data_token = []
	for st in data:
		data_token.append(get_token_text(st))
	return data_token
def get_vocab(data):
    #将分词放入set，不重复。类似建立语料库
	vocab = set(chain(*data))
	vocab_size = len(vocab)
    #建立语料库和索引
	word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'
	return vocab,vocab_size,word_to_idx,idx_to_word
def encode_st(token_data,vocab,word_to_idx):
	features = []
	for sample in token_data:
		feature = []
		for token in sample:
			if token in word_to_idx:
				feature.append(word_to_idx[token])
			else:
				feature.append(0)
		features.append(feature)
	return features
#填充和截断
#填充和截断
def pad_st(features,maxlen,pad=0):
    mask=[]
    padded_features = []
    for feature in features:
        if len(feature)>maxlen:
            mask.append(maxlen)
            padded_feature = feature[:maxlen]
        else:
            mask.append(len(feature))
            padded_feature = feature
            while(len(padded_feature)<maxlen):
                padded_feature.append(pad)
        padded_features.append(padded_feature)
    return padded_features,mask
data_path = "data/"
save_path = ""
maxlen = 600
train_data,train_label = read_file(data_path+"train/seq.in",data_path+"train/seq.out")
test_data,test_label = read_file(data_path+"test/seq.in",data_path+"test/seq.out")
#转化为小写
train_token = get_token_data(train_data)
test_token = get_token_data(test_data)
print("get_token_data success!")
vocab_t,vocab_size_t,word_to_idx_t,idx_to_word_t = get_vocab(train_token)
#np.save("vocab_t.npy",vocab_t)
vocab_l,vocab_size_l,word_to_idx_l,idx_to_word_l = get_vocab(train_label)
#np.save("vocab_l.npy",vocab_l)
print(idx_to_word_l)
print("vocab_save success!")
train_features,mask_train = pad_st(encode_st(train_token, vocab_t,word_to_idx_t),maxlen)
test_features,mask_test = pad_st(encode_st(test_token, vocab_t,word_to_idx_t),maxlen)
train_label,_ = pad_st(encode_st(train_label, vocab_l,word_to_idx_l),maxlen)
test_label,_ = pad_st(encode_st(test_label, vocab_l,word_to_idx_l),maxlen)
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, *args, **kwargs):
        self.data = [{'x':X[i],'y':Y[i]} for i in range(X.shape[0])]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
batch_size=256
train_features = torch.tensor(train_features)
test_features = torch.tensor(test_features)
train_labels = torch.tensor(train_label)
test_labels = torch.tensor(test_label)
train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=False)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)
epochs = 60
lr = 0.0005  # initial learning rate
lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
weight_decay = 1e-5  # 损失函数

embed_size = 300
hidden_dim = 600
dropout = 0.5
word2vec_path='data/glove_to_word2vec.txt'
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
weight = torch.zeros(vocab_size_t + 1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx_t[wvmodel.index2word[i]]
        #print("成功")

    except:
       # print("失败")
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word_t[word_to_idx_t[wvmodel.index2word[i]]]))
import torch.nn as nn
from torchcrf import CRF
def read(path):
    data=[]
    for line in open(path):
        data.append(line.strip().split(' '))
    return data
vail_data=read("data/conll03.txt")
vail_token = get_token_data(vail_data)
vail_features,mask_vail = pad_st(encode_st(vail_token, vocab_t,word_to_idx_t),maxlen)
vail_features = torch.tensor(vail_features)
vail_set = torch.utils.data.TensorDataset(vail_features, test_labels)
vail_iter = torch.utils.data.DataLoader(vail_set, batch_size=1,shuffle=False)
class NERLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_dim, weight, dropout, word2id, tag2id):
        super(NERLSTM_CRF, self).__init__()

        # self.embedding_dim = embedding_dim
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.vocab_size = len(word2id) + 1
        self.tag_to_ix = tag2id
        self.tagset_size = len(tag2id)

        # self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding.weight.requires_grad = True
        self.dropout = nn.Dropout(dropout)

        # CRF
        self.lstm = nn.LSTM(self.embed_size, self.hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=False)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        self.crf = CRF(self.tagset_size)

    def forward(self, x):
        # CRF
        x = x.transpose(0, 1)
        batch_size = x.size(1)
        sent_len = x.size(0)

        embedding = self.embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        # print(outputs.shape)
        # CRF
        outputs = self.crf.decode(outputs)
        return outputs

    def log_likelihood(self, x, tags):
        x = x.transpose(0, 1)
        batch_size = x.size(1)
        sent_len = x.size(0)
        tags = tags.transpose(0, 1)
        embedding = self.embedding(x)
        outputs, hidden = self.lstm(embedding)
        outputs = self.dropout(outputs)
        outputs = self.hidden2tag(outputs)
        # print(outputs.shape)
        return - self.crf(outputs, tags)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
best_f1=0
output_path="r1.txt"
def huanyuan(test):
    return [idx_to_word_l[i] for i in test[0]]
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
for i in range(10):
    net = NERLSTM_CRF(vocab_size=(vocab_size_t + 1), embed_size=embed_size, hidden_dim=hidden_dim, weight=weight,
                        dropout=dropout, word2id=word_to_idx_t, tag2id=word_to_idx_l)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    net = net.to(device)
    for epoch in range(epochs):
        f = open(output_path, "a")
        n = 0
        m = 0
        train_f1 = 0
        train_recall = 0
        test_f1 = 0
        test_recall = 0
        net.train()
        all_train_y=[]
        all_test_y=[]
        for x, y in train_iter:
            n += x.shape[0]
            #print("n",n)
            optimizer.zero_grad()
            net.zero_grad()
            loss = net.log_likelihood(x.to(device), y.to(device))
            #print("loss:",loss)
            predict = net.forward(x.to(device))
            #print("pre",predict)
            loss.backward()
            # CRF
            torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
            # print(predict,y)
            optimizer.step()
            #print("xxx")
            for z in predict:
                all_train_y.append(z)
        y = train_labels.numpy().tolist() 
        for ll in range(n):
            train_recall += recall_score(y[ll][:mask_train[ll]], all_train_y[ll][:mask_train[ll]], average='macro')
           # print("recall",recall)
            train_f1 += f1_score(y[ll][:mask_train[ll]], all_train_y[ll][:mask_train[ll]], average='macro')
        with torch.no_grad():
            net.eval()
            for x, y in test_iter:
                    m += x.shape[0]
                    predict = net.forward(x.to(device))
                   # print("test_pre",predict)
                    for z in predict:
                        all_test_y.append(z)
        y = test_labels.numpy().tolist() 
        for ll in range(m):
            test_recall += recall_score(y[ll][:mask_test[ll]], all_test_y[ll][:mask_test[ll]], average='macro')
           # print("recall",recall)
            test_f1 += f1_score(y[ll][:mask_test[ll]], all_test_y[ll][:mask_test[ll]], average='macro')
        f.write('epoch:%04d,-----train_f1:%f, train_recall:%f,test_f1:%f, test_recall:%f,best:%f\n' % (
        epoch, train_f1 / n, train_recall / n, test_f1 / m, test_recall / m,best_f1))
        f.close()
        if test_f1 / m > best_f1 and test_f1 / m >0.60:
            f2= open("end1.txt", "w+")
            best_f1=test_f1 / m
            #保存模型
            net.eval()
            with torch.no_grad():
                for x, y in vail_iter:
                    predict = net.forward(x.to(device))
                    f2.write(str(huanyuan(predict)))
                    f2.write('\n')
                f2.close()
        #训练到一定程度就跑路
        if train_f1 / n >0.998:
            break
        
