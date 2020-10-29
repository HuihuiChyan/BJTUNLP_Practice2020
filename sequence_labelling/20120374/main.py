from lstm_crf import BiLSTM_CRF
from seqeval.metrics import precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torchtext import data
from torchtext.vocab import Vectors
import os
import numpy as np
import torch.nn as nn

BATCH_SIZE = 1
EMBEDDING_DIM = 300
HIDDEN_DIM = 400
DROPOUT = 0.5
SENTENCE_MAX_LEN = 300
START_TAG = "<START>"
STOP_TAG = "<STOP>"
device = torch.device("cuda:5")
lr = 0.01
num_epochs = 50

if not os.path.exists('.vector_cache'):
    os.mkdir('.vector_cache')
vectors = Vectors(name='./glove.840B.300d.txt')



def tokenizer_text(text):
    return text.split()

TEXT = data.Field(sequential=True, stop_words=None, tokenize=tokenizer_text, lower=True,
                  batch_first=True)
LABEL = data.Field(sequential=True, tokenize=tokenizer_text, batch_first=True)
train_data, valid_data, test_data, = data.TabularDataset.splits(path='./',
                                                                train='train.csv',
                                                                validation='valid.csv',
                                                                test='final_test.csv',
                                                                format='csv',
                                                                skip_header=True,
                                                                fields=[('Text', TEXT), ('Label', LABEL)])


TEXT.build_vocab(train_data, vectors=vectors)
vocab_size = len(TEXT.vocab)
weight_matrix = TEXT.vocab.vectors

LABEL.build_vocab(train_data)
tag_to_ix = LABEL.vocab.stoi
tagset_size = len(tag_to_ix)+2
print(tag_to_ix)

train_iter, valid_iter = data.BucketIterator.splits((train_data, valid_data),
                                                    batch_size=BATCH_SIZE,
                                                    shuffle=True,
                                                    device=device,
                                                    sort_key=lambda x: len(x.Text)
                                                    )
test_iter = data.Iterator(test_data,
                          batch_size=BATCH_SIZE,
                          shuffle=False,
                          device=device,
                          sort=False,
                          repeat=False
                          )


def train(net, train_iter, valid_iter, device, num_epochs):
    train_loss, valid_loss = [], []
    train_batch_num = 0
    best_valid_acc = 0.
    for epoch in range(num_epochs):
        total_loss = 0.
        correct = 0
        sample_num = 0
        net.train()
        for context in train_iter:
            train_batch_num += 1
            data = context.Text
            data = data.to(device).long()
            target = context.Label
            target = target.to(device).long()
            masks = (data != TEXT.vocab.stoi["<pad>"])
            output = net(data, masks)
            optimizer.zero_grad()
            loss = net.loss(output, masks, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        v_loss,precision,recall,f1 = evaluate_accuracy(valid_iter, net)
        valid_loss.append(v_loss)
        loss = total_loss / train_batch_num
        train_loss.append(loss)
        if f1 > best_valid_acc:
            best_valid_acc = f1
            torch.save(net.state_dict(), "./best1.pth")
        print('epoch %d, train loss %.4f,  valid loss %.4f, valid precision %.4f, valid recall %.4f,valid f1 %.4f'
              % (epoch + 1, loss, v_loss,precision,recall,f1))
    return train_loss, valid_loss


def evaluate_accuracy(data_iter, net):
    acc_sum, valid_loss, n = 0.0, 0.0, 0
    valid_batch_num = 0
    net.eval()
    y_pred,y_true = [],[]
    for context in data_iter:
        valid_batch_num += 1
        X = context.Text
        X = X.to(device).long()
        y = context.Label
        y = y.to(device).long()
        masks = (X != TEXT.vocab.stoi["<pad>"])
        feats = net(X,masks)
        path_score, best_path = net.crf(feats, masks.bool())
        loss = net.loss(feats,masks,y)
        valid_loss += loss.item()
        n += y.shape[0]

        for seq_idx,mm in zip(best_path,masks):
            p = []
            for idx,m in zip(seq_idx,mm):
                if m == True:
                    p.append(LABEL.vocab.itos[idx])
            y_pred.append(p)

        for seq_idx,mm in zip(y,masks):
            p = []
            for idx,m in zip(seq_idx,mm):
                if m == True:
                    p.append(LABEL.vocab.itos[idx])
            y_true.append(p)
    print(y_pred[100],y_true[100])
    return valid_loss / valid_batch_num,precision_score(y_true, y_pred), recall_score(y_true, y_pred), f1_score(y_true, y_pred)

def test_acc(net, test_iter):
    net.load_state_dict(torch.load("/data/laisiyu/LSTM_CRF/best_83.92.pth"))
    net.eval()
    y_pred, y_true = [], []
    for context in test_iter:
        X = context.Text
        X = X.to(device).long()
        y = context.Label
        y = y.to(device).long()
        masks = (X != TEXT.vocab.stoi["<pad>"])
        feats = net(X,masks)
        path_score, best_path = net.crf(feats, masks.bool())

        for seq_idx,mm in zip(best_path,masks):
            p = []
            for idx,m in zip(seq_idx,mm):
                if m == True:
                    p.append(LABEL.vocab.itos[idx])
            y_pred.append(p)
        for seq_idx,mm in zip(y,masks):
            p = []
            for idx,m in zip(seq_idx,mm):
                if m == True:
                    p.append(LABEL.vocab.itos[idx])
            y_true.append(p)

    return y_pred,y_true

print("start")
net = BiLSTM_CRF(weight_matrix, vocab_size, tagset_size,EMBEDDING_DIM, HIDDEN_DIM, 1, DROPOUT, DROPOUT,device).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

train_loss, valid_loss = train(net, train_iter, valid_iter, device, num_epochs)
# pred,true= test_acc(net,test_iter)
# with open("./test.txt",'w',encoding='utf-8') as f:
#     for line in pred:
#         f.write(' '.join(line)+'\n')

# p,r,f = precision_score(true, pred), recall_score(true, pred), f1_score(true, pred)
# print('test precision %.4f, test recall %.4f,valid f1 %.4f'
#               % (p,r,f))

