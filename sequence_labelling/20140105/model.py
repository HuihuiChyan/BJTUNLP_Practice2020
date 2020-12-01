import torch
import torch.nn as nn
import pdb 

torch.manual_seed(1)

START_TAG = 'START'
STOP_TAG = 'STOP'

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):
    def __init__(self,vocab_size, tag_to_ix, embedding_dim, embedding_vector, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.word_embeds.weight.data.copy_(embedding_vector)
        self.word_embeds.weight.requires_grad = False
    
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, dropout=0.5, bidirectional=True)
                
        # 将LSTM的输出映射到标记空间
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # 转移参数矩阵 转移i到j的分数
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

        # 这两个语句强制执行了这样的约束，我们不会将其转移到开始标记，也不会将其转移到停止标记
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        # self.hidden = self.init_hidden()

    def init_hidden(self,batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).cuda(), 
        torch.randn(2, batch_size, self.hidden_dim // 2).cuda())

    def _forward_alg(self, feats):
        # 正向算法计算分块函数
        """
        feats (len_seq, batch_size, target_size)
        """
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -1000.).cuda() 
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.
        forward_var_list = []
        forward_var_list.append(init_alphas) 
        for feat_index in range(feats.shape[1]):  # -1
            forward_score = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            emit_score = torch.unsqueeze(feats[:,feat_index, :], 1).transpose(1, 2)  # +1
            total_score = forward_score + emit_score + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(total_score, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha
    
    def _get_lstm_features(self, sentence):
        batch_size = sentence.size(0)
        seq_length = sentence.size(1) # (seq_len, batch_size)(1,10)
        self.hidden = self.init_hidden(batch_size) # layer, batch_size, hidden_dim//2 (2, 1, 512)
        # pdb.set_trace()
        embeds = self.word_embeds(sentence).view(seq_length, batch_size, -1)
        
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # pdb.set_trace()
        lstm_out = lstm_out.view(batch_size, -1 , self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out) # (batch_size, seq_len,tagert_size)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # 给出所提供的标记序列的分数
        score = torch.zeros(tags.shape[0]).cuda()
        # pdb.set_trace()
        Start = torch.full([tags.shape[0],1], self.tag_to_ix[START_TAG],dtype=torch.long).cuda()
        tags = torch.cat((Start, tags), dim=1)
        
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            # pdb.set_trace()
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:,-1]]
        # pdb.set_trace()
        return score

    def _viterbi_decode(self, feats):
        """
        feats (seq_len,batch_size,target_size)
        """
        backpointers = []
        bptrs_t = [] 
        viterbivars_t = [] 
        
        # 在对数空间中初始化viterbi变量。
        init_vvars = torch.full((feats.shape[0], self.tagset_size), -10000.).cuda() 
        init_vvars[:,self.tag_to_ix[START_TAG]] = 0

        # 第i步的forward_var存放第i-1步的viterbi变量。
        forward_var = init_vvars
        for feat in range(feats.shape[1]):
            next_tag_var = torch.stack([forward_var] * feats.shape[2]).transpose(0, 1) + self.transitions
            viterbivars, best_tag_id = torch.max(next_tag_var, dim=2)
            viterbivars, best_tag_id = viterbivars.squeeze(0), best_tag_id.squeeze(0)
            bptrs_t.append(best_tag_id.cpu().numpy().tolist())
            viterbivars_t.append(viterbivars.cpu().detach().numpy().tolist())
            forward_var = (viterbivars + feats[:, feat, :])
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        path_score, best_tag_id = torch.max(terminal_var, dim=1)
        path_score, best_tag_id = path_score.squeeze(0).view(-1,1), best_tag_id.squeeze(0).view(-1,1).cpu().numpy().tolist()

        # 按照后面的来解码最佳路径
        best_path = [best_tag_id]
        for bptrs in reversed(torch.Tensor(bptrs_t).long()):
            best_tag_id = [bptrs[i][best_tag_id[i]].cpu().numpy().tolist() for i in range(feats.shape[0]) ]
            best_path.append(best_tag_id)
        
        best_path = torch.Tensor(best_path).long().permute(2,0,1).squeeze(0).cpu().numpy().tolist()
        start = best_path.pop()
        best_path.reverse()
        # print("best_path: ", torch.Tensor(best_path).long().transpose(1,0).cpu().numpy().tolist())
        return path_score, torch.Tensor(best_path).long().transpose(1,0).cpu().numpy().tolist()
    
    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        # pdb.set_trace()
        gold_score = self._score_sentence(feats, tags)
        # print('gold_score',torch.sum(forward_score - gold_score))
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        #Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        # print('forward====>',score, tag_seq)
        return score, tag_seq