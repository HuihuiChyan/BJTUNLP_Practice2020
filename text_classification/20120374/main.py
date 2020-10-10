from textcnn import text_cnn
import torch
import torch.nn as nn
from torchtext import data
from torchtext.vocab import Vectors
from torch.nn import init
# from torch.utils import data
from pretrain import prepocess,get_numpy_word_embed
import numpy as np
import os
import nltk
import tqdm
import spacy

batch_size =64
embedding_dim = 300
hidden_size = 128
n_filters = 200
filters_sizes = [2,3,4,5]
sentence_max_len = 400
output_dim=2
dropout=0.5
num_epochs = 50
device = torch.device("cuda:5")
lr = 0.0001

if not os.path.exists('.vector_cache'):
    os.mkdir('.vector_cache')
vectors = Vectors(name='./glove.840B.300d.txt')

def tokenizer(text):
    #return [tok.text for tok in spacy_en.tokenize(text)]
    return [tok for tok in nltk.word_tokenize(text)]

TEXT = data.Field(sequential=True,stop_words=None,tokenize=tokenizer,lower=True,fix_length=sentence_max_len,batch_first=True)
LABEL = data.Field(sequential=False,use_vocab=False,batch_first=True)



train_data,valid_data,test_data, = data.TabularDataset.splits(path = './',
                                                  train = 'train.csv',
                                                  validation='valid.csv',
                                                  test = 'test.csv',
                                                  format = 'csv',
                                                  skip_header=True,
                                                  fields = [('Text',TEXT),('Label',LABEL)])

TEXT.build_vocab(train_data,vectors = vectors)
vocab_size = len(TEXT.vocab)
weight_matrix = TEXT.vocab.vectors

train_iter, valid_iter= data.BucketIterator.splits((train_data,valid_data),
                                       batch_size = batch_size,
                                       shuffle = True,
                                       device = device,
                                       sort_key = lambda x:len(x.Text)
                                       )
test_iter = data.Iterator(test_data,
                                       batch_size = batch_size,
                                       shuffle = False,
                                       device = device,
                                       sort = False,
                                       repeat = False
                                       )

def evaluate_accuracy(data_iter, net):
    acc_sum, valid_loss ,n = 0.0, 0.0,0
    valid_batch_num = 0
    net.eval()
    for context in data_iter:
        valid_batch_num += 1
        X= context.Text
        X = X.to(device).long()
        y = context.Label
        y = y.to(device).long()
        y_hat = net(X)
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        loss = criterion(y_hat,y).sum()
        valid_loss += loss.item()
        n += y.shape[0]
    return acc_sum / n, valid_loss/valid_batch_num

def train(net,train_iter,valid_iter,device,num_epochs):
    train_loss, valid_loss =[],[]
    train_batch_num = 0
    best_valid_acc = 0.

    for epoch in range(num_epochs):
        total_loss = 0.
        correct = 0
        sample_num = 0
        net.train()
        for context in train_iter:
            train_batch_num += 1
            data= context.Text
            data =data.to(device).long()
            target = context.Label
            target= target.to(device).long()
            output = net(data)
            optimizer.zero_grad()
            loss = criterion(output,target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            prediction = torch.argmax(output,1)
            #print(prediction)
            correct +=(prediction ==target).sum().item()
            sample_num += len(prediction)
        valid_acc,v_loss= evaluate_accuracy(valid_iter,net)
        valid_loss.append(v_loss)
        loss = total_loss / train_batch_num
        train_loss.append(loss)
        acc = correct / sample_num
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            torch.save(net.state_dict(),"./best.pth")
        print('epoch %d, train loss %.4f, train acc %.3f, valid loss %.4f, valid acc %.3f'
              %(epoch + 1, loss, acc,v_loss,valid_acc))
    return train_loss, valid_loss

def test_acc(net, test_iter):
    net.load_state_dict(torch.load("./best.pth"))
    net.eval()
    pred = []
    for context in test_iter:
        X= context.Text
        X = X.to(device).long()
        y_hat = net(X)
        p = torch.argmax(y_hat,1)
        pred.append(p.tolist())
    return pred





criterion = torch.nn.CrossEntropyLoss()
net = text_cnn(vocab_size,weight_matrix,embedding_dim,hidden_size,n_filters,filters_sizes,sentence_max_len,output_dim,dropout).to(device)
optimizer = torch.optim.Adam(net.parameters(),lr=lr)

train_loss, valid_loss = train(net,train_iter,valid_iter,device,num_epochs)
pred = test_acc(net,test_iter)

with open("prediction2.txt",'w',encoding = 'utf8') as f:
    for line in pred:
        for l in line:
            f.write(str(l)+'\n')




