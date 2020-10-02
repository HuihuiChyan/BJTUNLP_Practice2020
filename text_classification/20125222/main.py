import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import numpy as np
import os
import time
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
def read_file(path):
	a = np.load(path,allow_pickle=True)
	a = a.tolist()
	return a

#参考博文https://cloud.tencent.com/developer/article/1491567、https://wenjie.blog.csdn.net/article/details/108436234
class textCNN(nn.Module):
	"""docstring for CNN"""
	def __init__(self, vocab_size,embed_size,seq_len,labels,weight,droput,**kwargs):
		super(textCNN, self).__init__(**kwargs)
		self.labels = labels
        #静态层embedding，不训练
		self.embedding_static = nn.Embedding(vocab_size,embed_size)
		self.embedding_static.weight.requires_grad = False
        #动态embedding，训练参数
		self.embedding_dynamic = nn.Embedding(vocab_size,embed_size)
		self.embedding_dynamic.weight.requires_grad = True
		self.conv1 = nn.Conv2d(2,1,(3,embed_size))
		self.conv2 = nn.Conv2d(2,1,(4,embed_size))
		self.conv3 = nn.Conv2d(2,1,(5,embed_size))
		self.pool1 = nn.MaxPool2d((seq_len - 3 + 1, 1))
		self.pool2 = nn.MaxPool2d((seq_len - 4 + 1, 1))
		self.pool3 = nn.MaxPool2d((seq_len - 5 + 1, 1))
		self.dropout = nn.Dropout(droput)
		self.linear = nn.Linear(3,labels)

	def forward(self,inputs):
		#print("inputs shape:",inputs.shape)
        #input为随机批量个数*截断长度（此时inputs还是文本形式，其实是文本的索引）
		inputs_1 = self.embedding_static(inputs).view(inputs.shape[0],1,inputs.shape[1],-1)
		inputs_2 = self.embedding_dynamic(inputs).view(inputs.shape[0],1,inputs.shape[1],-1)
		#print("inputs_1 shape:",inputs_1.shape)
		inputs = torch.cat((inputs_1,inputs_2),1)
        #embedding后，此时变为随机批量个数*输出通道数*截断长度*词向量长度
		#print("inputs shape:",inputs.shape)
		x1 = F.relu(self.conv1(inputs))
		#print("x1 shape:",x1.shape)
		x2 = F.relu(self.conv2(inputs))
		#print("x2 shape:",x2.shape)
		x3 = F.relu(self.conv3(inputs))
		#print("x3 shape:",x3.shape)
		x1 = self.pool1(x1)
		#print("x1 shape:",x1.shape)
		x2 = self.pool2(x2)
		#print("x2 shape:",x2.shape)
		x3 = self.pool3(x3)
		#print("x3 shape:",x3.shape)
		x = torch.cat((x1,x2,x3),1)
		#print("x shape:",x.shape)
		x = x.view(inputs.shape[0], 1, -1)
		#print("x shape:",x.shape)
		x = self.dropout(x)

		x = self.linear(x)
		#print("x shape:",x.shape)
		#logit = F.log_softmax(x, dim=1)
		x = x.view(-1, self.labels)
		#print("x:",x.shape)
		return(x)
embed_size = 300
num_hidens = 100
num_layers = 2
bidirectional = True
batch_size = 32
labels = 2
lr = 0.05
device = torch.device('cuda:6')


use_gpu = True
seq_len = 300
droput = 0.05
num_epochs = 100
train_features_path = "data1/test_features.npy"
test_features_path = "data1/test_features.npy"
test_label_path = "data1/test_label.npy"
train_label_path = "data1/train_label.npy"
vocab_path = "data1/vocab.npy"

word2vec_path = "data1/glove_to_word2vec.txt"
output_path = "data1/result.txt"
train_features = read_file(train_features_path)
test_features = read_file(test_features_path)
train_label = read_file(train_label_path)
test_label = read_file(test_label_path)
vocab = read_file(vocab_path)
vocab_size = len(vocab)

train_features = torch.tensor(train_features)
test_features = torch.tensor(test_features)
train_labels = torch.tensor(train_label)
test_labels = torch.tensor(test_label)

train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)
train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,shuffle=False)
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path,binary=False, encoding='utf-8')
weight = torch.zeros(vocab_size+1, embed_size)
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmode.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(idx_to_word[word_to_idx[wvmodel.index2word[i]]]))


net = textCNN(vocab_size=(vocab_size+1), embed_size=embed_size , seq_len= seq_len, labels=labels, weight=weight,droput=droput)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,3,5'
net = nn.DataParallel(net)
net = net.cuda()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)
#optimizer = optim.Adam(net.parameters(), lr=lr)


print("train start")
best_test_acc  = 0
for epoch in range(num_epochs):
    f=open(output_path,"a")
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    for feature, label in train_iter:
        n += 1
        net.zero_grad()
        feature = feature.cuda()
        label = label.cuda()
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,dim=1), label.cpu())
        train_loss += loss
    for test_feature, test_label in test_iter:
        m += 1
        test_feature = test_feature.cuda()
        test_label = test_label.cuda()
        test_score = net(test_feature)
        test_loss = loss_function(test_score, test_label)
        test_acc += accuracy_score(torch.argmax(test_score.cpu().data,dim=1), test_label.cpu())
        test_losses += test_loss
    end = time.time()
    runtime = end - start
    if(test_acc>best_test_acc and test_acc/m>0.86):
        best_test_acc = test_acc
        torch.save(net.state_dict(),'params_{}.pkl'.format(best_test_acc/m))
    f.write('epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f \n' %(epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,runtime))
    f.close()
    print('epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f' %(epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,runtime))


