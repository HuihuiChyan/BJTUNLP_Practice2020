from torchtext import data
from tkinter import _flatten
import pandas as pd
import torchtext
import torch as tc
# 处理数据
tag_to_idx = {'B-ORG': 0,'O': 1,'B-MISC': 2,'B-PER':3,'I-PER':4,'B-LOC': 5,'I-ORG': 6,'I-MISC': 7,'I-LOC': 8,'STOP':9,'START':10}
idx_to_tag = ['B-ORG','O','B-MISC','B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'STOP', 'START']
# 构建csv文件 
def process_file(base_path, tag_to_idx, Type):
    with open(base_path + Type + '/seq.in', 'r') as f_seq_in, \
        open(base_path + Type + '/seq.out', 'r') as f_seq_out:
        seq_lists = [seq.strip() for seq in f_seq_in.readlines()] 
        tags_lists = [[str(tag_to_idx[tag]) for tag in tags.strip().split()] for tags in f_seq_out.readlines()]
        print(tags_lists[0])
        tags_lists = [' '.join(tags) for tags in tags_lists]
        print(len(seq_lists))
        df_ = pd.DataFrame({'Seq':seq_lists, 'Tag': tags_lists, 'Char_': seq_lists})
        df_.to_csv('./Dataset/'+ Type + '.csv', index = False)
def get_csv_file(tag_to_idx):
    base_path = './Dataset/CoNLL2003_NER/'
    process_file(base_path, tag_to_idx, Type = 'train')
    process_file(base_path, tag_to_idx, Type = 'test')
    process_file(base_path, tag_to_idx, Type = 'valid')

def pad_char_list(char_list):
    max_len = max([len(item) for item in char_list])
    return [item + [1]*(max_len - len(item)) for item in char_list]
# 获取sentence数据迭代器
def get_data_iter():
    #获取字符vocab分词器
    def char_vocab_tokenizer(sentence):
        c_lists = [[c for c in word] for word in sentence.strip().split()]
        return list(_flatten(c_lists))
    def tag_tokenizer(x):
        rel = [int(tag) for tag in x.split()]
        return rel
    def _get_dataset(csv_data,char_to_idx, seq, tag, char_, char_len):
        examples = []
        fileds = [('Seq',seq),('Tag',tag),('Char_',char_),('Char_len',char_len)]
        for seq, tag in zip(csv_data['Seq'], csv_data['Tag']):
            char_list = [[char_to_idx[c] for c in word] for word in seq.strip().split()]
            char_len_list = [len(word) for word in seq.strip().split()]
            examples.append(data.Example.fromlist([seq, tag, pad_char_list(char_list),char_len_list], fileds))
        return examples, fileds
    
    seq = data.Field(sequential= True, use_vocab= True, lower= True)
    tag = data.Field(sequential= True, lower= False, use_vocab= False, tokenize= tag_tokenizer)
    char_ = data.Field(sequential=True, use_vocab = False, batch_first= True)
    char_len = data.Field(sequential=True, use_vocab=False,batch_first=True)
    char_vocab = data.Field(sequential=True, use_vocab = True, tokenize = char_vocab_tokenizer)  #只是用来构建字符集词典
    get_charvocab_fields=[('Seq',char_vocab),('None',None),('None',None)]
    train = data.TabularDataset.splits(path='./Dataset', train='train.csv',format='csv',skip_header=True,fields=get_charvocab_fields)[0]
    char_vocab.build_vocab(train) #字符集的词典
    # 构建Dataset数据集
    train_data = pd.read_csv('./Dataset/train.csv')
    val_data = pd.read_csv('./Dataset/valid.csv')
    test_data = pd.read_csv('./Dataset/test.csv')
    train_dataset = data.Dataset(*_get_dataset(train_data,char_vocab.vocab.stoi, seq, tag,char_, char_len))
    val_dataset = data.Dataset(*_get_dataset(val_data,char_vocab.vocab.stoi, seq, tag, char_, char_len))
    test_dataset = data.Dataset(*_get_dataset(test_data,char_vocab.vocab.stoi, seq, tag, char_, char_len))
    # 构造词典
    seq.build_vocab(train_dataset, vectors = torchtext.vocab.Vectors(name ='./Dataset/glove.6B.100d.txt'))
    # 构造数据迭代器    
    train_iter = data.BucketIterator(train_dataset, batch_size=1,shuffle=True ,sort_key = lambda x:len(x.Seq), device = tc.device('cpu'))
    val_iter, test_iter = data.BucketIterator.splits((val_dataset,test_dataset),batch_sizes=(1,1), shuffle=False,repeat=False,sort=False,device=tc.device('cpu'))
    return seq, char_vocab, train_iter, test_iter, val_iter
