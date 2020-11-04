import torch
import os
import pdb
import pickle
import nltk
from torch.utils import data
import csv
from torchtext import data
from torchtext.vocab import Vectors


# 将数据读入csv文件中
def read_data(path):
    with open(path+'/seq.in','r',encoding='utf-8') as fin,open(path+'/seq.out','r',encoding='utf-8') as fout:
        texts = [line.strip() for line in fin.readlines()]
        labels = [line.strip() for line in fout.readlines()]

    out_path = open(path + '.csv', 'w', encoding='utf-8')
    out_path.write('TEXT\tLABEL\n')
    for text,label in zip(texts,labels):
        out_path.write(text+'\t'+label+'\n')
    out_path.close()

# read_data('./data/train')
# read_data('./data/valid')
# read_data('./data/test')

# 分词器
def tokenizer(text):
    return text.split()

# 加载数据集，生成迭代器
def load_data(batch_size,device):
    # 文本
    text = data.Field(sequential=True, tokenize=tokenizer, lower = True, batch_first=True)
    # 标签
    label = data.Field(sequential=True, tokenize=tokenizer,batch_first=True)
    # 构建DataSet
    train, valid, test = data.TabularDataset.splits(
        path='./data/',
        skip_header=True,
        train='train.csv',
        validation = 'valid.csv',
        test='test.csv',
        format='csv',
        fields=[('TEXT', text), ('LABEL', label)],
    )
    # 创建词表
    text.build_vocab(train, vectors=Vectors(name='./data/glove.6B.300d.txt'))

    label.build_vocab(train)

    # 构建迭代器
    train_iter,valid_iter = data.BucketIterator.splits(datasets=(train,valid),
                                sort_key=lambda x: len(x.TEXT),
                                sort_within_batch=False,
                                shuffle=True,
                                batch_size=batch_size,
                                repeat=False,
                                device=device)

    test_iter = data.Iterator(test,
                         sort=False,
                         shuffle=False,
                         sort_within_batch=False,
                         batch_size=batch_size,
                         repeat=False,
                         device=device)

    return train_iter, valid_iter, test_iter, text.vocab
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# train_iter, valid_iter, test_iter, text_vocab = load_data(256,device)
# for batch in train_iter:
#     print(batch.TEXT)
#     print(batch.LABEL)
#     break



