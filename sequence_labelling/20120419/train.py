#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch as tc
from torch import nn
import pandas as pd
from torchtext import data
import torchtext
import time
import argparse
from torch import autograd
from torch.autograd import Variable
from tkinter import _flatten
import numpy as np
import pandas as pd
from seqeval.metrics import precision_score, recall_score, f1_score


# In[2]:


parser = argparse.ArgumentParser()
parser.add_argument('--lr',type=float, default = 0.01, help='学习率')
parser.add_argument('--save_path',type=str, default='./Model/model_.pth',help='模型保存位置')
parser.add_argument('--char_lstm_embed_size',type=int, default= 25 , help='字符集lstm嵌入dim')
parser.add_argument('--char_lstm_hidden_size',type=int, default= 25 , help='字符集sltm隐藏层dim')
parser.add_argument('--word_embed_size',type=int, default = 200, help='word嵌入dim')
parser.add_argument('--input_embed_size',type=int, default = 250, help='lstm_input_嵌入dim')
parser.add_argument('--hidden_size',type=int , default = 250, help='decoder_lstm隐藏层dim')
parser.add_argument('--add_dropout',type= int , default = 1, help='input_embed是否dropout')
parser.add_argument('--device',type=str , default ='cuda:2', help='train device')
args = parser.parse_args()
print(f'lr = {args.lr}')
print(f'save_path = {args.save_path}')
print(f'add_dropout = {args.add_dropout}')

def argmax(vec):
    # 返回行向量的最大值的index
    _, idx = tc.max(vec, dim = 1)
    return idx.item()
# 以数值稳定的方法计算路径概率分值
def log_sum_exp(vec):
    max_score = vec[0,argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + tc.log(tc.sum(tc.exp(vec - max_score_broadcast)))
# lstm初始化
def init_lstm(input_lstm):
    for ind in range(0, input_lstm.num_layers):
        weight = eval('input_lstm.weight_ih_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
        weight = eval('input_lstm.weight_hh_l' + str(ind))
        bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
        nn.init.uniform(weight, -bias, bias)
    if input_lstm.bidirectional:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.weight_ih_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)
            weight = eval('input_lstm.weight_hh_l' + str(ind) + '_reverse')
            bias = np.sqrt(6.0 / (weight.size(0) / 4 + weight.size(1)))
            nn.init.uniform(weight, -bias, bias)

    if input_lstm.bias:
        for ind in range(0, input_lstm.num_layers):
            weight = eval('input_lstm.bias_ih_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
            weight = eval('input_lstm.bias_hh_l' + str(ind))
            weight.data.zero_()
            weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
        if input_lstm.bidirectional:
            for ind in range(0, input_lstm.num_layers):
                weight = eval('input_lstm.bias_ih_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1
                weight = eval('input_lstm.bias_hh_l' + str(ind) + '_reverse')
                weight.data.zero_()
                weight.data[input_lstm.hidden_size: 2 * input_lstm.hidden_size] = 1


# In[3]:


#定义BILSTM-CRF模型
class BiLSTM_CRF(nn.Module):
    def __init__(self, tag_to_idx, vocab, char_vocab, args):
        super(BiLSTM_CRF, self).__init__()
        # char字符集embed层
        self.embedding1 = nn.Embedding(len(char_vocab.itos), args.char_lstm_embed_size)
        # word 单词集embed层
        self.embedding2 = nn.Embedding(len(vocab.itos), args.word_embed_size)
        self.vocab = vocab
        self.hidden_size = args.hidden_size
        self.tag_to_idx = tag_to_idx
        self.char_lstm_hidden_size = args.char_lstm_hidden_size
        self.target_size = len(tag_to_idx)
        # 定义输入前的drop_out操作
        self.drop_out = nn.Dropout(0.5)
        self.drop_or_not = args.add_dropout
        self.device = args.device
        #定义Bi-lstm层, lstm1:用于加载char级别词向量，lstm2用于decoder整体输入Input
        self.lstm1 = nn.LSTM(args.char_lstm_embed_size, args.char_lstm_hidden_size, num_layers=1, bidirectional = True)
        init_lstm(self.lstm1)
        self.lstm2 = nn.LSTM(args.input_embed_size, args.hidden_size, num_layers=1, bidirectional = True)
        # 定义全连接输出层
        self.decoder = nn.Linear(2 * args.hidden_size, self.target_size)
        # 定义转换矩阵
        self.transtitions = nn.Parameter(tc.rand(self.target_size,self.target_size))
        # 初始化start和end的转换矩阵data
        self.transtitions.data[tag_to_idx[START],:] = -10000
        self.transtitions.data[:, tag_to_idx[STOP]] = -10000 #从j->i为转移概率
    
    # 获取bi_lstm2的输出，对应的是k+2个标签的概率值
    def _get_lstm_features_decoder(self, sentence, char_, char_len):
        # sentence.shape: (seq_len，batch)  
        # char_.shape is (batch, seq_len, max_wordlen)
        # 获取char字符embedding
        char_len = char_len.squeeze(dim = 0) # shape: [seq_len]
        char_ = char_.squeeze(dim = 0).permute(1,0)
        char_embeddings = self.embedding1(char_)# shape : (max_wordlen, seq_len, embed_size),其中seq_len为batch
        outputs, _ = self.lstm1(char_embeddings) # # outputs_shape : (max_wordlen, seq_len, hidden_size)
        embeds = self.embedding2(sentence) # shape: (seq_len, batch, embed_size) 
        char_embeddings_process = Variable(tc.FloatTensor(tc.zeros((outputs.size(1),outputs.size(2))))).to(self.device)
        # char_embeddings_process: [seq_len, 2 * char_lstm_hidden_size]
        for i, index in enumerate(char_len):
            char_embeddings_process[i] = tc.cat((outputs[0][i,self.char_lstm_hidden_size:], outputs[index.cpu().item()-1][i,:self.char_lstm_hidden_size]))
        embeds = tc.cat((embeds.squeeze(dim= 1),char_embeddings_process), dim =1)
        embeds = embeds.unsqueeze(1)
        if self.drop_or_not:
            embeds = self.drop_out(embeds)
        outputs, _ = self.lstm2(embeds)
        outputs = outputs.permute(1,0,2) # shape:(batch,seq_len,hidden_size)
        outputs = self.drop_out(outputs)
        outputs = self.decoder(outputs)
        return outputs   # shape:(batch, seq_len,target_size)
    
    # 获取label_tags对应的概率分值
    def _get_gold_score(self, feats, tags):
        # # feats.shape: [seq_leb, target_size]
        # tags.shape: (batch, seq_len)
        temp = tc.LongTensor(range(feats.size()[0]))
        tags = tags.squeeze(0)
        add_start_tags = tc.cat([tc.LongTensor([self.tag_to_idx[START]]).to(self.device), tags])
        add_stop_tags = tc.cat([tags, tc.LongTensor([self.tag_to_idx[STOP]]).to(self.device)])
        gold_score =  tc.sum(self.transtitions[add_stop_tags, add_start_tags]) + tc.sum(feats[temp,tags])
        return gold_score
    
    # 计算所有路径的概率分值
    def _forward_alg(self, feats): 
        # feats.shape: [seq_leb, target_size]
        init_alphas = tc.Tensor(1, self.target_size).fill_(-10000.)
        init_alphas[0][self.tag_to_idx[START]] = 0.
        forward_var = autograd.Variable(init_alphas)
        forward_var = forward_var.to(self.device)
        for feat in feats:
            emit_score = feat.view(-1,1)
            tag_var = forward_var + self.transtitions + emit_score
            max_tag_var, _ = tc.max(tag_var, dim = 1) 
            tag_var = tag_var - max_tag_var.view(-1,1)
            forward_var = max_tag_var + tc.log(tc.sum(tc.exp(tag_var), dim=1)).view(1,-1)
        terminal_var = (forward_var + self.transtitions[self.tag_to_idx[STOP]]).view(1,-1)
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    # 获取字符列表
    def _get_char_list(self, sentence):
        str_list = [list(self.vocab.itos[index]) for index in sentence]
        return tc.tensor(str_list)
    # 计算误差值
    def _net_log_likelihood(self, sentence, tags, char_, char_len):
        # 输入，sentence(seq_len,batch)和真实标签(seq_len,batch)，char_(seq_len, batch)
        sentence = Variable(sentence)
        tags = Variable(tags)
        feats2 = self._get_lstm_features_decoder(sentence, char_, char_len)
        tags = tags.permute(1,0)
        feats2 = feats2.squeeze(0)
        forward_score = self._forward_alg(feats2)
        gold_score = self._get_gold_score(feats2, tags)
        return forward_score - gold_score
    # 预测真实标签
    def _viterbi_decode(self, feats):
        # feats.shape:(seq_len,target_size)
        backpointers = [] # 记录路径
        init_vvars = tc.full((1, self.target_size), -10000.).to(self.device)
        init_vvars[0][self.tag_to_idx[START]] = 0.
        forward_var = Variable(init_vvars)
        forward_var = forward_var.to(self.device)
        # 为何要使用list切换
        # 为何要使用list切换
        # 为何要使用list切换
        # 为何要使用list切换
        for feat in feats:
            next_tag_var = forward_var.view(1,-1).expand(self.target_size, self.target_size) + self.transtitions
            viterbivars_t, bptrs_t = tc.max(next_tag_var, dim=1)
            forward_var = viterbivars_t + feat
            backpointers.append(bptrs_t.tolist())
        terminal_var = forward_var + self.transtitions[self.tag_to_idx[STOP]]
        terminal_var.data[self.tag_to_idx[STOP]] = -10000.
        terminal_var.data[self.tag_to_idx[START]] = -10000.
        best_id = argmax(terminal_var.unsqueeze(dim=0))
        best_path = [best_id]
        for better_id in reversed(backpointers):
            best_id = better_id[best_id]
            best_path.append(best_id)
        start = best_path.pop()
        assert start == self.tag_to_idx[START]
        best_path.reverse()
        return best_path
    
    def forward(self,sentence, char_, char_len):
        # 输入，sentence(seq_len, batch)
        feats = self._get_lstm_features_decoder(sentence, char_, char_len) 
        tag_seq = self._viterbi_decode(feats.squeeze(0)) 
        return tag_seq


# In[4]:


# 处理数据
tag_to_idx = {'B-ORG': 0,'O': 1,'B-MISC': 2,'B-PER':3,'I-PER':4,'B-LOC': 5,'I-ORG': 6,'I-MISC': 7,'I-LOC': 8,'STOP':9,'START':10}
idx_to_tag = ['B-ORG','O','B-MISC','B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'STOP', 'START']
# 构建csv文件 
def process_file(base_path, tag_to_idx, Type):
    with open(base_path + Type + '/seq.in', 'r') as f_seq_in,         open(base_path + Type + '/seq.out', 'r') as f_seq_out:
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
    seq.build_vocab(train_dataset, vectors = torchtext.vocab.Vectors(name ='./Dataset/glove.6B.200d.txt'))
    # 构造数据迭代器    
    train_iter = data.BucketIterator(train_dataset, batch_size=1,shuffle=True ,sort_key = lambda x:len(x.Seq), device = tc.device('cpu'))
    val_iter, test_iter = data.BucketIterator.splits((val_dataset,test_dataset),batch_sizes=(1,1), shuffle=False,repeat=False,sort=False,device=tc.device('cpu'))
    return seq, char_vocab, train_iter, test_iter, val_iter


# In[5]:


def test_(net, data_iter, device, idx_to_tag):
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    seq_pred, seq_true = [],[]
    net.eval() # 进行测试模式
    for batch_data in data_iter:  
        sentence = (batch_data.Seq).to(device)
        tags = (batch_data.Tag).to(device)
        char_ = (batch_data.Char_).to(device)
        char_len = (batch_data.Char_len).to(device)
        loss = net._net_log_likelihood(sentence, tags, char_, char_len)
        tag_seq = net(sentence, char_, char_len)
        loss_sum += loss.cpu().item()
        n += sentence.shape[1]
        # 计算准确率
        true_seq = (tags.squeeze(1)).tolist()
        seq_pred.append(tag_seq)
        seq_true.append(true_seq)
        if n % 200 == 0:
            print(f'test__ n = {n}')
    net.train()  # 进入训练模式
    seq_pred = [[idx_to_tag[idx] for idx in seq_idx]for seq_idx in seq_pred]
    seq_true = [[idx_to_tag[idx] for idx in seq_idx]for seq_idx in seq_true]
    return loss_sum / n , precision_score(seq_true, seq_pred), recall_score(seq_true, seq_pred), f1_score(seq_true, seq_pred)
        
def train(net, num_epochs, train_iter, val_iter, test_iter ,optimizer, device, idx_to_tag):
    print(f'training on :{device}')
    min_num = 0.884
    for epoch in range(num_epochs):
        loss_sum, n, start,temp_time = 0.0, 0, time.time(), time.time()
        for batch_data in train_iter:
            sentence = (batch_data.Seq).to(device)
            tags = (batch_data.Tag).to(device)
            char_ = (batch_data.Char_).to(device)
            char_len = (batch_data.Char_len).to(device)
            loss = net._net_log_likelihood(sentence, tags, char_, char_len)         
            optimizer.zero_grad()
            loss.backward()
            # 进行梯度裁剪
            nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad, net.parameters()),5.0)
            optimizer.step()
            loss_sum += loss
            n += 1
            if n % 500 == 0:
                print(f'n = %d , train loss : %.3f time: %d' %(n, loss_sum / n, time.time()-temp_time))
                temp_time = time.time()
        loss , P, R, f1 = test_(net, val_iter, device, idx_to_tag)
        loss_test , P_test, R_test, f1_test = test_(net, test_iter, device, idx_to_tag)
        if f1_test >= min_num:
            min_num = f1_test
            print('Save model...')
            tc.save(net.state_dict() ,args.save_path)
        print('---->n = %d, Train loss : %.3f, val_loss: %.3f f1-score %.3f , Take time: %.3f'%(epoch, loss_sum / n, loss, f1, time.time()-start))
        print('-->Test: loss: %.3f pression: %.3f recall: %.3f  F1: %.3f'%(loss_test , P_test, R_test, f1_test))


# In[6]:


# 获取数据迭代器
seq, char_, train_iter, test_iter, val_iter = get_data_iter()
START ='START'
STOP = 'STOP'
device = tc.device('cuda:2')
print(f'device = {device}')
net = BiLSTM_CRF(tag_to_idx, seq.vocab, char_.vocab, args)
# 参数初始化
net.embedding2.weight.data.copy_(seq.vocab.vectors)
net.embedding2.weight.requires_grad = False
optim = tc.optim.SGD(filter(lambda p:p.requires_grad, net.parameters()), lr= args.lr, momentum = 0.9)
net.load_state_dict(tc.load(args.save_path))
net = net.to(device)
print(f'net.device is {net.device}')

# In[7]:


num_epochs = 50
train(net, num_epochs, train_iter, val_iter, test_iter, optim, device, idx_to_tag)

