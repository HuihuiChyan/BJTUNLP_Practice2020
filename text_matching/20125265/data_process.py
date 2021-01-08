from torch.utils import data
import torch
from torchtext import data
from torchtext.vocab import Vectors
import jsonlines
import csv
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import os

#改写缩略字
def replace_abbreviations(text):
    text = text.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's", "she is")\
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's", "that is")\
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't", "does not")\
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not")\
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's", "let us")\
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't", "could not")\
            .replace("you'll", "you will").replace("you've", "you have")
    return text

#去掉标点符号和数字
def remove(text):
    remove_chars = '[0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
    return re.sub(remove_chars, '', text)

#去停用词，并将动词词性恢复
def stemed_words(text):
    stop_words = stopwords.words("english")
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(w, pos='v') for w in text.split() if w not in stop_words]
    result = " ".join(words)
    return result

def process(text):
    text = replace_abbreviations(text)
    text = remove(text)
    text = stemed_words(text)
    return text

root = "/data/yinli/pytorch/task3/snli_1.0/"
# 将数据读入csv文件中
def read_data(f_in, f_out):
    out_file = open(os.path.join(root, f_out), 'w+', encoding='utf-8')
    label2id = {'neutral': 0, 'contradiction': 1, 'entailment': 2}
    writer = csv.writer(out_file)
    writer.writerow(('label', 'sentence1', 'sentence2'))
    with open(os.path.join(root, f_in),'r+',encoding='utf-8') as f:
        count = 0
        for item in jsonlines.Reader(f):
            label = item['gold_label']
            if label in label2id.keys():
                label = label2id[label]
            else:
                continue
            sen1 = item['sentence1']
            sen1 = process(sen1.lower())
            sen2 = item['sentence2']
            sen2 = process(sen2.lower())
            writer.writerow((label, sen1, sen2))
            count += 1
            print(count)
            if(count==150000):
                break
    out_file.close()

def read_test_data(f_in, f_out):
    out_file = open(os.path.join(root, f_out), 'w+', encoding='utf-8')
    writer = csv.writer(out_file)
    writer.writerow(('sentence1', 'sentence2'))
    with open(os.path.join(root, f_in),'r+',encoding='utf-8') as f:
        count = 0
        for sentence in f.readlines():
            sentence = sentence.split("|||")
            sen1 = sentence[0]
            sen2 = sentence[1]
            sen1 = process(sen1.lower())
            sen2 = process(sen2.lower())
            writer.writerow((sen1, sen2))
            count += 1
            print(count)
    out_file.close()

#read_data("snli_1.0_train.jsonl", "train5.csv")
# read_data("snli_1.0_dev.jsonl", "dev2.csv")
#read_test_data("snli.test", "test3.csv")

#分词器
def tokenizer(text):
    return text.strip().split()
#
# 加载数据集，生成迭代器
def load_data(batch_size, device):
    # 标签
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)
    # 文本
    SEN1 = data.Field(sequential=True, tokenize=tokenizer,  fix_length=50, lower=True, batch_first=True)
    SEN2 = data.Field(sequential=True, tokenize=tokenizer,  fix_length=50, lower=True, batch_first=True)

    # 构建DataSet
    train, valid = data.TabularDataset.splits(
        path='./snli_1.0/',
        skip_header=True,
        train="train4.csv",
        validation="dev3.csv",
        format='csv',
        fields=[("label", LABEL), ("sentence1", SEN1), ("sentence2", SEN2)],
    )

    test = data.TabularDataset(
        path='./snli_1.0/test3.csv',
        skip_header=True,
        format='csv',
        fields=[("sentence1", SEN1), ("sentence2", SEN2)],
    )

    # 创建词表
    SEN1.build_vocab((train.sentence11, train.sentence2), vectors=Vectors(name='/data/yinli/dataset/glove.840B.300d.txt'))
    SEN2.vocab = SEN1.vocab

    # 构建迭代器
    train_iter = data.BucketIterator(train,
                                sort_key=lambda x: len(x.SEN1),
                                sort_within_batch=False,
                                shuffle=True,
                                batch_size=batch_size,
                                repeat=False,
                                device=device)

    valid_iter = data.Iterator(valid,
                              sort=False,
                              shuffle=False,
                              sort_within_batch=False,
                              batch_size=batch_size,
                              repeat=False,
                              train=False,
                              device=device)

    test_iter = data.Iterator(test,
                               sort=False,
                               shuffle=False,
                               sort_within_batch=False,
                               batch_size=batch_size,
                               repeat=False,
                               train=False,
                               device=device)

    return train_iter, valid_iter, test_iter, SEN1.vocab, SEN2.vocab

# 加载数据集，生成迭代器
# def load_data(batch_size, device):
#     # 标签
#     LABEL = data.Field(sequential=True, batch_first=True)
#     # 文本
#     SEN1 = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
#     SEN2 = data.Field(sequential=True, tokenize=tokenizer, lower=True, batch_first=True)
#
#     # 构建DataSet
#     train = data.TabularDataset(
#         path='./snli_1.0/train2.csv',
#         skip_header=True,
#         format='csv',
#         fields=[("label", LABEL), ("sentence1", SEN1), ("sentence2", SEN2)],
#     )
#
#     # 创建词表
#     SEN1.build_vocab(train, vectors=Vectors(name='/data/yinli/dataset/glove.840B.300d.txt'))
#     SEN2.build_vocab(train, vectors=Vectors(name='/data/yinli/dataset/glove.840B.300d.txt'))
#     LABEL.build_vocab(train)
#
#     # 构建迭代器
#     train_iter = data.BucketIterator(train,
#                                 sort_key=lambda x: len(x.SEN1),
#                                 sort_within_batch=False,
#                                 shuffle=True,
#                                 batch_size=batch_size,
#                                 repeat=False,
#                                 device=device)
#
#     return train_iter, SEN1.vocab, SEN2.vocab


# device = torch.device("cuda:1")
# train_iter, dev_iter, test_iter, sentence1_vocab, sentence2_vocab = load_data(5, 50, device)
#
# for batch in train_iter:
#     print(batch.label)
#     print(batch.sentence1)
#     print(batch.sentence2)
#     break
# print(len(sentence1_vocab.vectors))
#
# print(sentence1_vocab.stoi['frown'])
# print(sentence2_vocab.stoi['frown'])
# print(sentence1_vocab.stoi['<unk>'])
#
# del train_iter
# del dev_iter
# del test_iter
# del sentence1_vocab
# del sentence2_vocab

#
# embedding = torch.cat((sentence2_vocab.vectors ,sentence1_vocab.vectors[2:]), 0)
# print(embedding.size())
# vocab_size, embed_size = embedding.size()
# print(vocab_size)
# print(embed_size)
# print(len(label_vocab))
# print(label_vocab.stoi)
#label2id = {'<unk>': 0, '<pad>': 1, 'neutral': 2, 'contradiction': 3, 'entailment': 4}