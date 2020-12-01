import os
import torch
import torch.nn as nn
import argparse
from collections import defaultdict,Counter
import torchtext.vocab as Vectors
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import pdb 
from torch.utils.data import Dataset, DataLoader
from model import BiLSTM_CRF
import numpy
from sklearn.metrics import classification_report

# 参数列表
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=17000)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--cuda', type=str, default='cuda:5')
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--param_path',type=str, default='./dataset/param.bin')
parser.add_argument('--test_path',type=str, default='./dataset/test/seq.in')
parser.add_argument('--test_result_path',type=str, default='./result.txt')
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()


device = torch.device(args.cuda if torch.cuda.is_available() else "cpu")
print(device)

def read_file(feature_path,label_path):
    """
    分词 feature转化为小写
    """
    feature = []
    label = []
    with open(feature_path) as seq_in:
        for line in seq_in.readlines():
            feature.append(line.strip().lower().split(' '))
    with open(label_path) as seq_out:
        for line in seq_out.readlines():
            label.append(line.strip().split(' '))
    return feature, label

def get_vocab(train_feature):
    if os.path.exists('./dataset/vocab.txt'):
        with open('./dataset/vocab.txt',"r",encoding='utf-8') as fvocab:
            vocab_words = [line for line in fvocab]
    else:
        train_word = []
        for line in train_feature:
            train_word.extend(line)
        counter = Counter(train_word)
        common_words = counter.most_common()
        print('common_words=======>',len(common_words))
        vocab_words = [word[0] for word in common_words[:args.vocab_size-2]]
        vocab_words = ['[UNK]','[PAD]'] + vocab_words
        with open("./dataset/vocab.txt","w",encoding='utf-8') as fvocab:
                for word in vocab_words:
                    fvocab.write(word+'\n')
    return vocab_words

word2vec_path = "../wvmodel/word2vec.txt"
def get_wvmodel():
    if os.path.exists(word2vec_path):
        print('gensim loading word2vec')
        # 使用gensim载入word2vec词向量
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
    else:
        # 已有的glove词向量
        glove_file = '../glove/glove.6B.300d.txt'
        # 指定转化为word2vec格式后文件的位置
        tmp_file = word2vec_path
        #glove词向量转化为word2vec词向量的格式
        glove2word2vec(glove_file, tmp_file)
        wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
    return wvmodel

def get_weight(wvmodel,vocab_size,embedding_size):
    """"
    去词向量文件中查表，得到词表中单词对应的权重weight
    """
    weight = torch.zeros(vocab_size, embedding_size)
    for i in range(len(wvmodel.index2word)):
        try:
            index = word2id[wvmodel.index2word[i]]
        except:
            continue
        weight[index, :] = torch.from_numpy(wvmodel.get_vector(id2word[word2id[wvmodel.index2word[i]]]))
    return weight

class SentenceDataSet(Dataset):
    def __init__(self, sent, sent_label):
        self.sent = sent
        self.sent_label = sent_label

    def __getitem__(self, item):
        return torch.Tensor(self.sent[item]), torch.Tensor(self.sent_label[item])

    def __len__(self):
        return len(self.sent)

def get_data(sample_features, sample_labels):
    sample_data = []                                                    #为了能够将data放到DataLoader中
    for i in range(len(sample_features)):
        temp = []
        temp.append(sample_features[i])
        temp.append(sample_labels[i])
        sample_data.append(temp)
    return sample_data

def collate_fn(sample_data):
    sample_data.sort(key=lambda data: len(data[0]), reverse=True)                          #倒序排序
    sample_features, sample_labels = [], []
    for data in sample_data:
        sample_features.append(data[0])
        sample_labels.append(data[1])
    data_length = [len(data[0]) for data in sample_data]                                   #取出所有data的长度             
    sample_features = torch.nn.utils.rnn.pad_sequence(sample_features, batch_first=True, padding_value=padding_value) 
    return sample_features, sample_labels, data_length

# 模型

def train(args, train_loader, eval_loader, model,optim,criterion):
    """
    训练函数
    """
    print("=============begin train=============")
    loss_log = []
    global_step = 0
    best_f1 = 0.0
    for epoch in range(args.epoch_num):
        total_acc,total_loss,correct,sample_num= 0, 0, 0,0
        for feature, batch_labels, data_length in train_loader:
            # feature (seq_len,batch_size) [1,10]
            feature = feature.cuda()

            label = [line.numpy().tolist() for line in batch_labels]
            for line in label:
                for i in range(feature.shape[1]-len(line)):
                    line.append(line[len(line)-1])
            # pdb.set_trace()
            label = torch.tensor(label).cuda()

            model.train()
            optim.zero_grad()
            
            loss = model.neg_log_likelihood(feature, label)

            loss.backward()
            optim.step()
            global_step += 1
            total_loss += loss.item()
            loss_log.append(loss.item())
        if global_step % args.steps_per_log == 0:
            print('Train {:d}| Loss:{:.5f}'.format(epoch+1 ,total_loss / len(train_loader)))
        if global_step % args.steps_per_eval == 0:
            test_loss, f1_min = evaluate_accuracy(model, eval_loader)
            print('at train step %d, eval loss is %.4f, eval f1 is %.4f' % (global_step, test_loss, f1_min))

            if f1_min > best_f1:
                best_f1 = f1_min
                torch.save(model.state_dict(), args.param_path)
                print("save model......")

def processing_len(out, label, batch_seq_len):
    out_pred = out[:batch_seq_len[0]]
    out_true = label[:batch_seq_len[0]]
    for i in range(1,len(batch_seq_len)):
        out_pred = torch.cat((out_pred,out[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i]]),dim=0)
        out_true = torch.cat((out_true,label[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i]]),dim=0)

    return out_pred.data.cpu().numpy(), out_true.data.cpu().numpy()


def evaluate_accuracy(model, test_loader):
    global_step,total_loss = 0, 0.0
    out_epoch, label_epoch = [], []
    with torch.no_grad():
        for feature, label, seq_len in test_loader:
            feature = feature.cuda()
            label = [line.numpy().tolist() for line in label]

            for line in label:
                for i in range(feature.shape[1]-len(line)):
                    line.append(line[len(line)-1])

            _, y_hat = model(feature)

            label = torch.tensor(label).cuda()

            model.eval()
            

            loss = model.neg_log_likelihood(feature, label)

            global_step += 1
            total_loss += loss.item()
            #测试集评价指标

            label = torch.tensor(label).view(-1,1).squeeze(-1).cuda()    
            out = torch.tensor(y_hat).view(-1,1).squeeze(-1).cuda()
            out, label = processing_len(out, label, seq_len)  

            out_epoch.extend(out)
            label_epoch.extend(label)
        report = classification_report(label_epoch, out_epoch, output_dict=True)
        # pdb.set_trace()
        print(report['macro avg']['f1-score'],report['macro avg']['recall'])
        return total_loss / global_step, report['macro avg']['f1-score']



train_feature, train_label = read_file('./dataset/train/seq.in','./dataset/train/seq.out')
valid_feature, valid_label = read_file('./dataset/valid/seq.in','./dataset/valid/seq.out')
test_feature, test_label = read_file('./dataset/test/seq.in','./dataset/test/seq.out')

# vocab_words = get_vocab(train_feature)
wvmodel = get_wvmodel()


# 缺省值 [UNK] 填充值 [PAD]
word2id = dict(zip(wvmodel.index2word,range(len(wvmodel.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(wvmodel.index2word)}     # id -> word
word2id['[UNK]'] = len(word2id)    
word2id['[PAD]'] = len(word2id)                          
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词
#获得标签字典
label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}
id2label = dict((idx, tag) for tag, idx in label2id.items())
# 词表生成 end

if args.train_or_test == "train":

    # 转化为词表里面的index
    train_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature]
    valid_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in valid_feature]
    test_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in test_feature]
    
    train_label = [[label2id[word] for word in line] for line in train_label]  
    valid_label = [[label2id[word] for word in line] for line in valid_label]
    test_label = [[label2id[word] for word in line] for line in test_label]

    train_textlines = [torch.Tensor(line).long() for line in train_textlines]
    train_label = [torch.Tensor(line).long() for line in train_label]
    valid_textlines = [torch.Tensor(line).long() for line in valid_textlines]
    valid_label = [torch.Tensor(line).long() for line in valid_label]
    test_textlines = [torch.Tensor(line).long() for line in test_textlines]
    test_label = [torch.Tensor(line).long() for line in test_label]


    train_data = get_data(train_textlines, train_label)
    valid_data = get_data(valid_textlines, valid_label)
    test_data = get_data(test_textlines, test_label)

    train_loader = torch.utils.data.DataLoader(dataset=train_data,
        batch_size=args.batch_size,collate_fn=collate_fn, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size,
        collate_fn=collate_fn, shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size,
                                               collate_fn=collate_fn, shuffle=False)

    weight_matrix = get_weight(wvmodel,len(word2id),args.embedding_size)
    print('weight_matrix',weight_matrix.size())
    model = BiLSTM_CRF(len(word2id),label2id, args.embedding_size, weight_matrix, args.hidden_size).cuda()
    
    if os.path.exists(args.param_path):
        print('loading params')
        # pdb.set_trace()
        model.load_state_dict(torch.load(args.param_path))

    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, train_loader,valid_loader, model, optim, criterion)
    end_loss, end_f1 = evaluate_accuracy(model, test_loader)
    print("====================>test loss: %.4f, test f1 : %.4f"%(end_loss, end_f1))
else:
    print('test begin')
    with open(args.test_path, 'r', encoding='utf-8') as ftest_text:
        test_textlines = [line.strip().lower().split(' ') for line in ftest_text.readlines()]

        test_textlines = [[word2id[word] if word in word2id else unk for word in line] for line in test_textlines]

        test_textlines = [torch.Tensor(line).long() for line in test_textlines]
        
        weight_matrix = get_weight(wvmodel,len(word2id),args.embedding_size)
        model = BiLSTM_CRF(len(word2id),label2id, args.embedding_size, weight_matrix, args.hidden_size).cuda()
    
        if os.path.exists(args.param_path):
            model.load_state_dict(torch.load(args.param_path))
            print('model initialized from params.bin')
        else:
            raise Exception('no params.bin?')

        model.eval()

        all_test_tag = []
        for idx in range(0, len(test_textlines), args.batch_size):

            batch_feature = test_textlines[idx:idx + args.batch_size]

            pad_batch_feature = torch.nn.utils.rnn.pad_sequence(batch_feature, batch_first=True, padding_value=padding_value)

            feature = pad_batch_feature.cuda()
            
            _, y_hat = model(feature)

            for idx , temp_hat in enumerate(y_hat):
                tags_temp = []
                for j in range(len(temp_hat)):
                    tags_temp.append(id2label[temp_hat[j]])
                all_test_tag.append(tags_temp[:len(batch_feature[idx])])


        with open(args.test_result_path, 'w', encoding='utf-8') as fresult:
            for tags_line in all_test_tag:
                for tag_word in tags_line:
                    fresult.write(tag_word + ' ')
                fresult.write('\n')
            print('test end')




