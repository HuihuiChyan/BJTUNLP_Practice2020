#!/usr/bin/env python
# coding: utf-8

from collections import Counter
import torch
import argparse
from gensim.models import Word2Vec
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import os

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:5')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-1)
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--ckp', type=str, default='./ckp/model_4.pt')
parser.add_argument('--f1_min', type=float, default=0.642816)
args = parser.parse_args()

if torch.cuda.is_available():
    print("using cuda")
    device = torch.device(args.cuda)

# ======================================= 加载数据集并处理 ======================================= #

#读入训练集
with open('./CoNLL2003_NER/train/seq.in', encoding='utf-8') as ftrain_feature:
    train_feature_line = [line.strip() for line in ftrain_feature.readlines()]
with open('./CoNLL2003_NER/train/seq.out', encoding='utf-8') as ftrain_label:
    train_label_line = [line.strip() for line in ftrain_label.readlines()]
#读入验证集
with open('./CoNLL2003_NER/test/seq.in', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
with open('./CoNLL2003_NER/test/seq.out', encoding='utf-8') as ftest_label:
    test_label_line = [line.strip() for line in ftest_label.readlines()]

#转换大小写并用split分隔开存入列表
train_feature_line = [line.lower().split(" ") for line in train_feature_line]
train_label_line = [line.split(" ") for line in train_label_line]
test_feature_line = [line.lower().split(" ") for line in test_feature_line]
test_label_line = [line.split(" ") for line in test_label_line]

#获得单词字典
w2v_model = gensim.models.KeyedVectors.load_word2vec_format('./w2v_model/word2vec_model_2.txt',binary=False, encoding='utf-8')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
unk = word2id['[UNK]']              #UNK:低频词
padding_value = word2id['[PAD]']    #PAD:填充词
#获得标签字典
label2id = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}

#获得数据和标签序列
train_feature = [[word2id[word] if word in word2id else unk for word in line] for line in train_feature_line]
train_label = [[label2id[word] for word in line] for line in train_label_line]
test_feature = [[word2id[word] if word in word2id else unk for word in line] for line in test_feature_line]
test_label = [[label2id[word] for word in line] for line in test_label_line]

#转成Tensor的形式
train_feature = [torch.Tensor(line).long() for line in train_feature]
train_label = [torch.Tensor(line).long() for line in train_label]
test_feature = [torch.Tensor(line).long() for line in test_feature]
test_label = [torch.Tensor(line).long() for line in test_label]

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

train_data = get_data(train_feature, train_label)
test_data = get_data(test_feature, test_label)

#处理非定长序列
train_dataloader = torch.utils.data.DataLoader(train_data, args.batch_size, collate_fn=collate_fn, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_data, args.batch_size, collate_fn=collate_fn, shuffle=True)


# ======================================= 模型定义 ======================================= #
class BiLSTM_CRF(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector, label2id):
        super(BiLSTM_CRF,self).__init__()
        
        # ============================ BiLSTM的系列参数 ============================ #
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=args.num_layers, 
                                    batch_first=True, dropout=0.5, bidirectional=True)
        self.out = torch.nn.Linear(2 * hidden_size,output_size)
        
        # ============================= CRF的系列参数 ============================= #
        self.tagset_size = len(label2id)
        self.tag_to_ix = label2id
        #转移矩阵，transitions[i][j]表示从label j转移到label i的概率
        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size,self.tagset_size))
        #这两个语句强制执行了这样的约束条件：我们永远不会转移到开始标签，也永远不会从停止标签转移
        self.transitions.data[self.tag_to_ix['START'], :] = -1000     #从任何标签转移到START_TAG不可能
        self.transitions.data[:, self.tag_to_ix['STOP']] = -1000      #从STOP_TAG转移到任何标签不可能

# ======================================= 模型前向传播 ======================================= #
    def forward(self, x, batch_seq_len):
        
        lstm_feats = self.BiLSTM(x, batch_seq_len) #batch_seq_len是每个batch中的序列的真实长度
        score, tag_seq = self.CRF(lstm_feats)      #BiLSTM处理结果作为CRF的输入,输出为分数和预测的标签序列
        return score, tag_seq

# ======================================= BiLSTM部分 ======================================= #
    def BiLSTM(self, x, batch_seq_len):
        batch_size = x.size(0)
        seq_len = x.size(1)
        x = self.embedding(x)
        
        h = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(x.device) 
        c = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(x.device) 
        
        #pack_padded_sequence和pad_packed_sequence可以避免很多不必要的计算
        x = torch.nn.utils.rnn.pack_padded_sequence(x,batch_seq_len, batch_first=True)
        output, hidden = self.bilstm(x, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        lstm_feats = self.out(output)     
        return lstm_feats
     
# ======================================= CRF的forward部分 ======================================= #
    def CRF(self, feats):
    # 具体解释：看CRF_forward.png笔记
        backpointers = []
        bptrs_t = []  
        viterbivars_t = []  

        #        我  是  梁  棋  棋
        #        |   |   |   |   |         这部分对应 for next_tag in range(self.tagset_size):
        # START->O ->O ->B ->I ->I ->STOP  这部分对应 self.transitions

        #每个time_step都需要前一个time_step的分数，每个time_step 对应 for feat in feats:
        #                                          分数score 对应 forward_var
        #所以forward_var = (torch.cat(viterbivars_t) + feat)

        forward_var = torch.full([feats.shape[0], self.tagset_size], -1000)
        forward_var[:, self.tag_to_ix['START']] = 0
        for feat in range(feats.shape[1]):
            next_tag_var = torch.stack([forward_var] * feats.shape[2]).transpose(0, 1) + transitions
            viterbivars, best_tag_id = torch.max(next_tag_var, dim=2)
            viterbivars, best_tag_id = viterbivars.squeeze(0), best_tag_id.squeeze(0)
            bptrs_t.append(best_tag_id.numpy().tolist())
            viterbivars_t.append(viterbivars.numpy().tolist())
            forward_var = (viterbivars + feats[:, feat, :])
            backpointers.append(bptrs_t)

        #   棋
        #   |
        #   I ->STOP 这部分对应 terminal_var
        #因为没有字对应‘STOP’标签，所以只需要加上转移到STOP的分数即可，即self.transitions[self.tag_to_ix['STOP']]

        terminal_var = forward_var + transitions[self.tag_to_ix['STOP']]
        path_score, best_tag_id = torch.max(terminal_var, dim=1)
        path_score, best_tag_id = path_score.squeeze(0).view(-1,1), best_tag_id.squeeze(0).view(-1,1).numpy().tolist()
        
        # 根据动态规划，由最后的节点，向前选取最佳的路径
        best_path = [best_tag_id]
        for bptrs in reversed(torch.Tensor(bptrs_t).long()):
            best_tag_id = [bptrs[i][best_tag_id[i]].numpy().tolist() for i in range(feats.shape[0]) ]
            best_path.append(best_tag_id)
        
        best_path = torch.Tensor(best_path).long().permute(2,0,1).squeeze(0).numpy().tolist()
        start = best_path.pop()
        best_path.reverse()
        print("best_path: ", torch.Tensor(best_path).long().transpose(1,0).numpy().tolist())
        return torch.Tensor(best_path).long().transpose(1,0).numpy().tolist()
    
# ======================================= 计算模型loss ======================================= #    
    def loss_function(self, sentence, tags, batch_seq_len):
        
        feats = self.BiLSTM(sentence, batch_seq_len)
        # loss = log(∑ e^s(X,y)) - s(X,y) 
        forward_score = self._forward_alg(feats)           # loss的log部分的结果
        gold_score = self._score_sentence(feats, tags)     # loss的S(X,y)部分的结果
        return torch.sum(forward_score - gold_score)

# ======================================= 计算loss的log部分 ======================================= #     
    def _forward_alg(self, feats):
    # 具体解释：看CRF_loss.png笔记
    
        # 加入batch的信息
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -1000.).to(device)     #保证了一定是从START到其他标签
        init_alphas[:, self.tag_to_ix['START']] = 0.                                        #START的位置变成0

        forward_var_list = []
        forward_var_list.append(init_alphas)   
            
        #        我  是  梁  棋  棋
        #        |   |   |   |   |         ...... emit_score   e[][]
        # START->O ->O ->B ->I ->I ->STOP  ...... trans_score  t[][]

        # 第一轮 feat：
        # 由于没有字对应START，所以forward_var = init_alphas 假设标签数只有2
        # “我” 转移到 0 的分数 S11 = t[START][0] + e[我][0]; 
        # “我” 转移到 1 的分数 S12 = t[START][1] + e[我][1]
        # log_sum_exp(next_tag_var)：log(e^S11 + e^S12)                                            
        
        # 第二轮 feat：
        # “是” 转移到 0 的分数 S21 = log( e^( “我” 转移到 0 的分数 + t[0][0] + e[是][0]) + 
        #                               e^( “我” 转移到 1 的分数 + t[1][0] + e[是][0])
        # “是” 转移到 0 的分数 S22 = log( e^( “我” 转移到 0 的分数 + t[0][1] + e[是][1]) + 
        #                               e^( “我” 转移到 1 的分数 + t[0][1] + e[是][1]) + 
        # log_sum_exp(next_tag_var)：log(e^S21 + e^S22) 
        #                          = log(e^( “我” 转移到 0 的分数 + t[0][0] + e[是][0]) + e^( “我” 转移到 1 的分数 + t[1][0] + e[是][0])
        #                               +e^( “我” 转移到 0 的分数 + t[0][1] + e[是][1]) + e^( “我” 转移到 1 的分数 + t[0][1] + e[是][1])
        #                          = log(e^S1 + e^S2 + e^S3 + e^S4) Si表示第i条路径
        for feat_index in range(feats.shape[1]):  # -1
            forward_score = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            emit_score = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            total_score = forward_score + emit_score + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(total_score, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix['STOP']].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha
            
    
# ======================================= 计算loss的S(X,y)部分 ======================================= #    
    def _score_sentence(self, feats, tags):
        #_forward_alg   ：算所有路径，找到一个最大可能路径，实际上可能不是真实标签转移的值
        #_score_sentence：直接用真实标签转移的值计算
        tags = torch.tensor(tags).to(device)
        score = torch.zeros(tags.shape[0]).to(device)
        Start = torch.full([tags.shape[0],1], self.tag_to_ix['START']).long().to(device)
        # 将'START'的标签拼接到 tag 序列最前面，表示开始
        tags = torch.cat((Start, tags), dim=1)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.tag_to_ix['STOP'], tags[:,-1]]
        return score

# ======================================= 模型测试 ======================================= #

#通过batch_seq_len留下out和label中不重复的部分进行指标计算
def processing_len(out, label, batch_seq_len):
    out_pred = out[:batch_seq_len[0]]
    out_true = label[:batch_seq_len[0]]
    for i in range(1,len(batch_seq_len)):
        out_pred = torch.cat((out_pred,out[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i]]),dim=0)
        out_true = torch.cat((out_true,label[i*batch_seq_len[0]:i*batch_seq_len[0]+batch_seq_len[i]]),dim=0)

    return out_pred.data.cpu().numpy(), out_true.data.cpu().numpy()

f1_min = args.f1_min
target_names = ['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']
def test_evaluate(model, test_dataloader, batch_size):
    test_l, n = 0.0, 0
    out_epoch, label_eppch = [], []
    global f1_min
    model.eval()
    with torch.no_grad():
        for data_x, data_y, batch_seq_len in test_dataloader:
            _, out = model(data_x.to(device),batch_seq_len)                #out就是路径序列 [10, 40]

            label = [line.numpy().tolist() for line in data_y]
            for line in label:
                for i in range(data_x.shape[1]-len(line)):
                    line.append(line[len(line)-1])
            loss = model.loss_function(data_x.to(device), label, batch_seq_len)

            for line in out:
                for i in range(data_x.shape[1]-len(line)):
                    line.append(line[len(line)-1])
            label = torch.tensor(label).view(-1,1).squeeze(-1).to(device)        #torch.Size([274])
            out = torch.tensor(out).view(-1,1).squeeze(-1).to(device)            #torch.Size([274])
            out, label = processing_len(out, label, batch_seq_len)   
            
            #测试集评价指标
            out_epoch.extend(out)
            label_eppch.extend(label)
            test_l += loss.item()
            n += 1

        print(classification_report(label_eppch, out_epoch, target_names=target_names, digits=6))
        report = classification_report(label_eppch, out_epoch, output_dict=True)
        if report['macro avg']['f1-score'] > f1_min : 
            f1_min = report['macro avg']['f1-score']
            torch.save(model.state_dict(), args.ckp)
            print("save model......")
            
        return test_l/n

# ======================================= 模型训练 ======================================= #

# 让Embedding层使用训练好的Word2Vec权重
if os.path.exists('word2vec_model.txt'):
    print("loading word2vec_model......")
    w2v_model = Word2Vec.load('word2vec_model.txt')
embedding_matrix = w2v_model.wv.vectors
input_size = embedding_matrix.shape[0]   
hidden_size = embedding_matrix.shape[1]  
model = BiLSTM_CRF(input_size, hidden_size, 11, embedding_matrix,label2id).to(device)
if os.path.exists(args.ckp):
    print("loading model......")
    model.load_state_dict(torch.load(args.ckp))
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay ) #
for epoch in range(args.num_epoch):
    model.train()
    train_l, n = 0.0, 0
    out_epoch, label_eppch = [], []

    start = datetime.datetime.now()
    for data_x, data_y, batch_seq_len in train_dataloader:
        _, out = model(data_x.to(device),batch_seq_len)                      #out就是路径序列 [10, 40]

        label = [line.numpy().tolist() for line in data_y]
        for line in label:
            for i in range(data_x.shape[1]-len(line)):
                line.append(line[len(line)-1])
        loss = model.loss_function(data_x.to(device), label, batch_seq_len)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        for line in out:
            for i in range(data_x.shape[1]-len(line)):
                line.append(line[len(line)-1])
        label = torch.tensor(label).view(-1,1).squeeze(-1).to(device)        #torch.Size([274])
        out = torch.tensor(out).view(-1,1).squeeze(-1).to(device)            #torch.Size([274])
        out, label = processing_len(out, label, batch_seq_len)        

        out_epoch.extend(out)
        label_eppch.extend(label)

        #训练集评价指标
        train_l += loss.item()
        n += 1

    #print(classification_report(label_eppch, out_epoch, target_names=target_names))
    test_loss = test_evaluate(model, test_dataloader, args.batch_size)
    end = datetime.datetime.now()

    print('epoch %d, train_loss %f, train_loss %f, fi_max %f, time %s'% (epoch+1, train_l/n, test_loss, f1_min, end-start))