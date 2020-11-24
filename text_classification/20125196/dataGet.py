import torch
import numpy as np
from itertools import chain
from collections import defaultdict
import argparse
import torch.optim as optim
import re
import os
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from sacremoses import MosesTokenizer

comments_test,comments_train,comments_valid = [],[],[]
labels_test,labels_train,labels_valid = [],[],[]

def file_list_name(path):
    file_list_name = []
    for file_name in os.listdir(path):
        file_list_name.append(path+file_name)
    return file_list_name

train_comments_pos = file_list_name("aclImdb/train/pos/")
train_comments_neg = file_list_name("aclImdb/train/neg/")

test_comments_pos = file_list_name("aclImdb/test/pos/")
test_comments_neg = file_list_name("aclImdb/test/neg/")

valid_comments = file_list_name("aclImdb/val/")

def comments_participle(text):

    text = text.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's", "she is")\
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's", "that is")\
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't", "does not")\
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not")\
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's", "let us")\
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't", "could not")\
            .replace("you'll", "you will").replace("you've", "you have")

    mt = MosesTokenizer(lang='en')
    text = [mt.tokenize(text, return_str=True)]
    text = [line.lower() for line in text]

    return text

# count=0
for f_name in train_comments_pos:
    # count=count+1
    with open(f_name,'r+', encoding='utf-8') as f:
        comments_train = comments_train+comments_participle(str(f.readlines()))
        labels_train = labels_train+[1]
        f.close()
    # if(count==100):
    #     break

# count=0
for f_name in train_comments_neg:
    # count = count + 1
    with open(f_name,'r+', encoding='utf-8') as f:
        comments_train = comments_train+comments_participle(str(f.readlines()))
        labels_train = labels_train+[0]
        f.close()
    # if(count==100):
    #     break

# count=0
for f_name in test_comments_pos:
    # count = count + 1
    with open(f_name,'r+', encoding='utf-8') as f:
        comments_test = comments_test+comments_participle(str(f.readlines()))
        labels_test = labels_test+[1]
        f.close()
    # if(count==100):
    #     break

# count=0
for f_name in test_comments_neg:
    # count=count+1
    with open(f_name,'r+', encoding='utf-8') as f:
        comments_test = comments_test+comments_participle(str(f.readlines()))
        labels_test = labels_test+[0]
        f.close()
    # if(count==100):
    #     break

for f_name in valid_comments:
    temp=[]
    with open(f_name, 'r+', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            comments_valid = comments_valid+comments_participle(line)

        for i in range(0,len(comments_valid)):
            temp= temp+[0]

        labels_valid=temp

comments_total = comments_train

def vocab_set():
    set_vocab = open('vocab.txt', 'w+', encoding='utf-8')
    vocab = set(comments_total)
    vocab_list=[]
    word_to_idx={}
    idx_to_word={}

    set_vocab.write('<unk>'+'\n')
    set_vocab.write('[PAD]' + '\n')
    vocab = list(vocab)
    for element in vocab:
        vocab_list = vocab_list+ list(element.split())

    vocab_list = set(vocab_list)
    vocab_list = list(set(vocab_list))
    for i in vocab_list:
        set_vocab.write(str(i)+'\n')
    set_vocab.close()

    i=2
    for word in vocab_list:
        word_to_idx[word]=i
        i=i+1
    word_to_idx['<unk>'] = 0
    word_to_idx['[PAD]'] = 0

    i=2
    for word in vocab_list:
        idx_to_word[i]=word
        i=i+1
    idx_to_word[0] = '<unk>'
    idx_to_word[1] = '[PAD]'

    vocab_size = len(vocab_list)+2

    return vocab_list, vocab_size, word_to_idx, idx_to_word

vocab_words,vocab_size, word_to_idx, idx_to_word = vocab_set()

def encode_set(comments, word_to_idx):
    features = []
    for sample in comments:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


comments_train = [str(line).split()[:300] for line in comments_train]
comments_test = [str(line).split()[:300] for line in comments_test]
comments_valid = [str(line).split()[:300] for line in comments_valid]

comments_train = [line + ['[PAD]' for i in range(300 - len(line))] for line in comments_train]
comments_test = [line + ['[PAD]' for i in range(300 - len(line))] for line in comments_test]
comments_valid = [line + ['[PAD]' for i in range(300 - len(line))] for line in comments_valid]

comments_train = encode_set(comments_train,word_to_idx)
comments_test = encode_set(comments_test, word_to_idx)
comments_valid = encode_set(comments_valid, word_to_idx)

comments_train = torch.tensor(comments_train)
labels_train = torch.tensor(labels_train)

comments_test = torch.tensor(comments_test)
labels_test = torch.tensor(labels_test)

comments_valid = torch.tensor(comments_valid)
labels_valid = torch.tensor(labels_valid)

train_dataset = torch.utils.data.TensorDataset(comments_train, labels_train)
test_dataset = torch.utils.data.TensorDataset(comments_test, labels_test)
valid_dataset = torch.utils.data.TensorDataset(comments_valid, labels_valid)

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=8000)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--input_channel', type=int, default=1)
parser.add_argument('--output_channel', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--output_size', type=int, default=2)
parser.add_argument('--kernal_sizes', type=list, default=[2,3,4,5])
args = parser.parse_args()

args.vocab_size = vocab_size

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=True)

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,batch_size=args.batch_size,shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        self.embedding_size = args.embedding_size
        self.vocab_size = args.vocab_size
        self.input_channel = args.input_channel
        self.output_channel = args.output_channel
        self.kernal_sizes = args.kernal_sizes
        self.output_size = args.output_size
        self.drop_out = args.dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.convs = nn.ModuleList(
            [nn.Conv2d(self.input_channel, self.output_channel, (k, self.embedding_size)) for k in self.kernal_sizes]
        )
        self.dropout = nn.Dropout(self.drop_out)
        self.linear = nn.Linear(len(self.kernal_sizes) * self.output_channel, self.output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        out = self.linear(x)
        return out

model = TextCNN(args).to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(100):
    correct1,correct2,sample_num1,sample_num2,train_acc, test_acc= 0,0,0,0,0,0
    train_loss,test_loss,test_losses,= 0,0,0
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        prediction = pred.argmax(dim=1)
        loss = criterion(pred, batch_y)

        correct1 += (prediction == batch_y).sum().item()
        sample_num1 += len(prediction)
        train_acc = correct1 / sample_num1
        train_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    with torch.no_grad():
        model.eval()
        for test_feature, test_label in test_loader:
            test_feature = test_feature.to(device)
            test_label = test_label.to(device)
            test_score = model(test_feature)
            predict = test_score.argmax(dim=1)
            test_loss = criterion(test_score, test_label)

            correct2 += (predict == test_label).sum().item()
            sample_num2 += len(predict)
            test_acc = correct2 / sample_num2
            test_losses += test_loss

    print('train_epoch:', '%d' % (epoch + 1), 'train_loss =', '{:.6f}'.format(train_loss.data/sample_num1),'train_acc =', '%.6f' % (train_acc),
              'test_loss =', '{:.6f}'.format(test_losses.data/sample_num2),'test_acc =', '%.6f' % (test_acc))

    if(test_acc>0.880000):
        testResults = open('testResults.txt', 'w+', encoding='utf-8')
        torch.save(model,'test.pth')
        model.eval()
        for x, y in valid_loader:
            valid_x = x.to(device)
            valid_y = y.to(device)
            valid_score = model(valid_x)
            p = valid_score.argmax(dim=1)
            testResults.write(str(p))

testResults.close()



































