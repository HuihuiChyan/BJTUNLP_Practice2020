import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from crf import CRF
from torch.autograd import Variable

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
    _, idx = torch.max(vec,1)
    return idx.item()



def log_sum_exp(vec):
    max_score_vec = torch.max(vec, dim=1)[0]
    max_score_broadcast = max_score_vec.view(vec.shape[0], -1).expand(vec.shape[0], vec.size()[1])
    return max_score_vec + torch.log(torch.sum(torch.exp(vec - max_score_broadcast), dim=1))


class BiLSTM_CRF(nn.Module):
    def __init__(self, weight_matrix, vocab_size,tagset_size, embedding_dim, hidden_dim, rnn_layers, dropout_ratio, dropout1,device,
                 use_cuda=True):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.word_embeds = nn.Embedding(vocab_size,embedding_dim)
        self.word_embeds.weight.data.copy_(weight_matrix)
        self.word_embeds.weight.requires_grad = False
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            num_layers=rnn_layers, bidirectional=True, dropout=dropout_ratio, batch_first=True)
        self.rnn_layers = rnn_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim * 2, tagset_size + 2)
        self.tagset_size = tagset_size
        self.device = device


    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable
        """
        return Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).to(self.device), Variable(
            torch.randn(2 * self.rnn_layers, batch_size, self.hidden_dim)).to(self.device)
    def forward(self, sentence, attention_mask=None):
        '''
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state
        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)
        embeds = self.word_embeds(sentence)
        hidden = self.rand_init_hidden(batch_size)
        # if embeds.is_cuda:
        # hidden = (i.cuda() for i in hidden)
        # print(hidden.type())
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim * 2)
        d_lstm_out = self.dropout1(lstm_out)
        l_out = self.liner(d_lstm_out)
        lstm_feats = l_out.contiguous().view(batch_size, seq_length, -1)
        return lstm_feats

    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value








