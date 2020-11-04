import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import pdb
from torch.autograd import Variable

START_TAG = "START"
STOP_TAG = "STOP"

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp_bacth(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))

class BiLSTM_CRF(nn.Module):

    def __init__(self, args, label2idx, weight,device):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.vocab_size = args.vocab_size
        self.tag_to_ix = label2idx
        self.tagset_size = len(label2idx)
        self.batch_size = args.batch_size
        self.device = device

        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.word_embeds.weight.data.copy_(weight)
        self.word_embeds.weight.requires_grad = False

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # 将BiLSTM提取的特征向量映射到特征空间，通过全连接得到发射矩阵
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # 初始化转移矩阵，transitions[i,j]代表第j个tag转移到第i个tag的转移分数
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size).to(self.device))

        # 其他转移到START的分数，以及STOP转移到其他的分数都非常小
        self.transitions.data[self.tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, self.tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden(args.batch_size)

    # 初始化lstm参数
    def init_hidden(self,batch_size):
        return (torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device),
                torch.randn(2, batch_size, self.hidden_dim // 2).to(self.device))

    # lstm获取特征
    def _get_lstm_features(self, sentence):
        # sentence: batch, seq_len
        batch_size = sentence.size(0)
        seq_len = sentence.size(1)
        self.hidden = self.init_hidden(batch_size)  # 2,batch,hidden_dim/2   2 256 256

        embeds = self.word_embeds(sentence) # batch,seq_len,embedding_dim    256 558 300

        lstm_out, self.hidden = self.lstm(embeds, self.hidden) # batch seq_len hidden_size

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        # 获得发射分数
        lstm_feats = self.hidden2tag(lstm_out).contiguous().view(batch_size,seq_len,-1)
        return lstm_feats

    # 计算给定的tag序列的分数，即一条序列的分数
    def _score_sentence(self, feats, tags):
        score_list = []
        for feat, tag in zip(feats,tags):
            score = torch.zeros(1).to(self.device)
            start_tag = torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).to(self.device)
            tag = torch.cat([start_tag,tag])
            for i, small_feat in enumerate(feat):
                score = score + self.transitions[tag[i + 1], tags[i]] + small_feat[tag[i + 1]]
            score = score + self.transitions[self.tag_to_ix[STOP_TAG], tag[-1]]
            score_list.append(score)
        pdb.set_trace()
        return torch.cat(score_list)

    # 前向算法：递推计算所有可能路径的分数组合——动态规划
    def _forward_alg(self, feats):
        batch_size = feats.shape[0]
        init_alphas = torch.full((batch_size, self.tagset_size), -10000.).to(self.device)
        # 初始化第0步，START位置的发射分数，START取0，其他都取-10000
        init_alphas[:,self.tag_to_ix[START_TAG]] = 0.

        # 赋给forward_var
        forward_var = autograd.Variable(init_alphas)
        forward_var = forward_var.to(self.device)
        convert_feats = feats.permute(1,0,2)

        # 迭代整个句子
        for feat in convert_feats:
            alphas_t = []  # 当前时间步的前向tensor
            # 分别计算各个tag的分数
            for next_tag in range(self.tagset_size):
                # 取出当前tag的发射分数，与之前时间步的tag无关
                emit_score = feat[:,next_tag].view(batch_size, -1).expand(batch_size, self.tagset_size)
                # 取出当前tag由之前tag转移过来的转移分数
                trans_score = self.transitions[next_tag].view(1, -1).repeat(batch_size,1)
                # 当前路径得分：之前时间步的分数+转移分数+发射分数
                next_tag_var = forward_var + trans_score + emit_score
                # 对当前分数进行log_sum_exp()
                alphas_t.append(log_sum_exp_bacth(next_tag_var))
            # 更新forward_var
            forward_var = torch.stack(alphas_t).permute(1, 0)
        # 考虑最终转移到STOP
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(1,-1).repeat(batch_size,1)
        #最终得分
        alpha = log_sum_exp_bacth(terminal_var)
        return alpha

        # alpha_list = []
        # for feats in batchfeats:
        #     # Do the forward algorithm to compute the partition function
        #     init_alphas = torch.full((1, self.tagset_size), -10000.)
        #     # START_TAG has all of the score.
        #     init_alphas[0][self.tag_to_ix[START_TAG]] = 0.
        #
        #     # Wrap in a variable so that we will get automatic backprop
        #     forward_var = autograd.Variable(init_alphas)
        #     forward_var = forward_var.to(self.device)
        #
        #     # Iterate through the sentence
        #     for feat in feats:
        #         alphas_t = []  # The forward tensors at this timestep
        #         for next_tag in range(self.tagset_size):
        #             # broadcast the emission score: it is the same regardless of
        #             # the previous tag
        #             emit_score = feat[next_tag].view(
        #                 1, -1).expand(1, self.tagset_size)
        #             # the ith entry of trans_score is the score of transitioning to
        #             # next_tag from i
        #             trans_score = self.transitions[next_tag].view(1, -1)
        #             # The ith entry of next_tag_var is the value for the
        #             # edge (i -> next_tag) before we do log-sum-exp
        #             next_tag_var = forward_var + trans_score + emit_score
        #             # The forward variable for this tag is log-sum-exp of all the
        #             # scores.
        #             alphas_t.append(log_sum_exp(next_tag_var).view(1))
        #         forward_var = torch.cat(alphas_t).view(1, -1)
        #     terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        #     alpha = log_sum_exp(terminal_var)
        #     alpha_list.append(alpha.view(1))
        #     pdb.set_trace()
        return torch.cat(alpha_list)

    # viterbi解码——每个时间步存的是：当前时间步每一个tag对应的之前的最优路径
    def _viterbi_decode(self, feats_list):
        path_list = []
        for feats in feats_list:
            backpointers = []  # 回溯指针

            # 初始化
            init_vvars = torch.full((1, self.tagset_size), -10000.).to(self.device)
            init_vvars[0][self.tag_to_ix[START_TAG]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            #forward_var = init_vvars
            forward_var = autograd.Variable(init_vvars)
            forward_var = forward_var.to(self.device)

            for feat in feats:
                bptrs_t = []  # 保存当前时间步的回溯指针
                viterbivars_t = []  # 保存当前时间步的viterbi变量
                # 对各个tag进行循环
                for next_tag in range(self.tagset_size):
                    # 记录最优路径时只考虑上一步的分数以及上一步tag转移到当前tag的转移分数
                    # 不取决于当前tag的发射分数
                    next_tag_var = forward_var + self.transitions[next_tag]
                    best_tag_id = argmax(next_tag_var) # 取最优
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) # 存最优路径
                # 更新forward_var ，加上当前tag的发射分数
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                # 记录当前时间步各个tag来源前一步的tag
                backpointers.append(bptrs_t)

            # 考虑转移到STOP_TAG
            terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]  # 最优路径对应的分数

            # 利用回溯指针解码出最优路径
            best_path = [best_tag_id]
            # best_tag_id作为开始，反向便利backpointers找到最优路径
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # 去除START_TAG
            start = best_path.pop()
            assert start == self.tag_to_ix[START_TAG]  # 检查
            best_path.reverse()
            path_list.append(best_path)
        return path_score,path_list

    # crf损失函数：真实路径的分数 和 所有路径的总分数
    # 真实路径的分数——应该是所有路径中分数最高的
    # log真实路径的分数 / log所有路径的总分数 ，越大越好，反过来，loss越小越好
    def neg_log_likelihood(self, sentence, tags):

        feats = self._get_lstm_features(sentence) # 发射分数
        forward_score = self._forward_alg(feats)  # 所有可能的路径分数
        gold_score = self._score_sentence(feats, tags)  # 标记好的tag对应的分数——真实路径的分数
        #pdb.set_trace()
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # 获得发射分数
        lstm_feats = self._get_lstm_features(sentence)

        # 找到最优路径，以及对应的分数
        score, tag_seq_list = self._viterbi_decode(lstm_feats)
        return score, tag_seq_list
