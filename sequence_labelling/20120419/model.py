import torch as tc
from torch import nn
import numpy as np
from torch.autograd import Variable
from torch import autograd
tag_to_idx = {'B-ORG': 0,'O': 1,'B-MISC': 2,'B-PER':3,'I-PER':4,'B-LOC': 5,'I-ORG': 6,'I-MISC': 7,'I-LOC': 8,'STOP':9,'START':10}
idx_to_tag = ['B-ORG','O','B-MISC','B-PER', 'I-PER', 'B-LOC', 'I-ORG', 'I-MISC', 'I-LOC', 'STOP', 'START']
START ='START'
STOP = 'STOP'
device = tc.device('cuda:2')

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