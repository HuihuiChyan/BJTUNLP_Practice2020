import torch

from torch import nn

class VariationalDropout(nn.Dropout):
    def forward(self, input_tensor):
        ones = input_tensor.data.new_ones(input_tensor.shape[0], input_tensor.shape[-1])
        dropout_mask = torch.nn.functional.dropout(ones, self.p, self.training, inplace=False)
        if self.inplace:
            input_tensor *= dropout_mask.unsqueeze(1)
            return None
        else:
            return dropout_mask.unsqueeze(1) * input_tensor

class EmbeddingLayer(nn.Module):
    """Implement embedding layer. """
    def __init__(self, embedding, dropout=0.5):
        super(EmbeddingLayer, self).__init__()
        vocab_size, vector_size = embedding.size()
        self.vector_size = vector_size
        self.embed = nn.Embedding(vocab_size, vector_size)
        self.embed.weight.data.copy_(embedding)
        self.dropout = VariationalDropout(dropout)

    def load(self, vectors):
        """Load pre-trained embedding weights."""
        self.embed.weight.data.copy_(vectors)

    def forward(self, x):
        e = self.embed(x)
        return self.dropout(e)

class EncodingLayer(nn.Module):
    """BiLSTM encoder which encodes both the premise and hypothesis."""
    def __init__(self, input_size, hidden_size):
        super(EncodingLayer, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers=1,bidirectional=True)

    def forward(self, x):#[batch, seq_len, input_size]
        self.lstm.flatten_parameters()
        output, _ = self.lstm(x)
        return output #[batch, seq_len, 2 * hidden_size]

class LocalInferenceModel(nn.Module):
    """The local inference model """
    def __init__(self):
        super(LocalInferenceModel, self).__init__()
        self.softmax_1 = nn.Softmax(dim=1)
        self.softmax_2 = nn.Softmax(dim=2)

    def forward(self, p, h, p_mask, h_mask):
        # p,h[batch, seq_len_p, 2 * hidden_size]  p_mask,h_mask[batch, seq_len], 0 in the mask
        # equation 11
        e = torch.matmul(p, h.transpose(1, 2))  # [batch, seq_len_p, seq_len_h]

        # masking the scores for padding tokens
        inference_mask = torch.matmul(p_mask.unsqueeze(2).float(), h_mask.unsqueeze(1).float())
        e.masked_fill_(inference_mask < 1e-7, -1e7)

        # equation 12 & 13
        h_score, p_score = self.softmax_1(e), self.softmax_2(e)
        h_ = h_score.transpose(1, 2).bmm(p)
        p_ = p_score.bmm(h)

        # equation 14 & 15
        m_p = torch.cat((p, p_, p - p_, p * p_), dim=-1)
        m_h = torch.cat((h, h_, h - h_, h * h_), dim=-1)

        assert inference_mask.shape == e.shape
        assert p.shape == p_.shape and h.shape == h_.shape
        assert m_p.shape[-1] == p.shape[-1] * 4

        return m_p, m_h #[batch, seq_len, 8 * hidden_size]


class CompositionLayer(nn.Module):
    """The composition layer. """
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        super(CompositionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.F = nn.Linear(input_size, output_size)
        self.lstm = nn.LSTM(output_size, hidden_size, num_layers=1, bidirectional=True)
        self.dropout = VariationalDropout(dropout)

    def forward(self, m):#[batch, seq_len, input_size]
        y = self.dropout(self.F(m))
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(y)#[batch, seq_len, hidden_size*2]

        assert m.shape[:2] == outputs.shape[:2] and \
            outputs.shape[-1] == self.hidden_size * 2
        return outputs #[batch, seq_len, hidden_size * 2]


class Pooling(nn.Module):
    """Apply maxing pooling and average pooling to the outputs of LSTM. """
    def __init__(self):
        super(Pooling, self).__init__()

    def forward(self, x, x_mask):#x[batch, seq_len, hidden_size * 2]，x_mask[batch, seq_len]
        mask_expand = x_mask.unsqueeze(-1).expand(x.shape)

        # average pooling
        x_ = x * mask_expand.float()
        v_avg = x_.sum(1) / x_mask.sum(-1).unsqueeze(-1).float()

        # max pooling
        x_ = x.masked_fill(mask_expand == 0, -1e7)
        v_max = x_.max(1).values

        assert v_avg.shape == v_max.shape == (x.shape[0], x.shape[-1])
        return torch.cat((v_avg, v_max), dim=-1)#[batch, hidden_size * 4]

class InferenceComposition(nn.Module):
    """Inference composition"""
    def __init__(self, input_size, output_size, hidden_size, dropout=0.5):
        super(InferenceComposition, self).__init__()
        self.composition = CompositionLayer(input_size, output_size, hidden_size, dropout=dropout)
        self.pooling = Pooling()

    def forward(self, m_p, m_h, p_mask, h_mask):
        #m_p,m_h[batch, seq_len, input_size],mask[batch, seq_len], 0 means padding
        # equation 16 & 17
        v_p, v_h = self.composition(m_p), self.composition(m_h)#[batch, hidden_size * 2]
        # equation 18 & 19
        v_p_, v_h_ = self.pooling(v_p, p_mask), self.pooling(v_h, h_mask)#[batch, hidden_size * 4]
        # equation 20
        v = torch.cat((v_p_, v_h_), dim=-1)

        assert v.shape == (m_p.shape[0], v_p.shape[-1] * 4)
        return v #[batch, input_size * 8]

class LinearSoftmax(nn.Module):
    """Implement the final linear layer."""
    def __init__(self, input_size, output_size, class_num, activation='relu', dropout=0.5):
        super(LinearSoftmax, self).__init__()
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError("Unknown activation function!!!")
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            self.dropout,
            nn.Linear(input_size, output_size),
            self.activation,
            nn.Linear(output_size, class_num),
        )

    def forward(self, x):#[batch, features]
        logits = self.fc(x)
        return logits #[batch, class_num]


class ESIM(nn.Module):
    def __init__(self, hidden_size, embedding=None, num_labels=3, dropout=0.5, device='gpu'):
        super(ESIM, self).__init__()
        self.device = device
        vocab_size, vector_size = embedding.size()
        self.embedding_layer = EmbeddingLayer(embedding, dropout)
        self.encoder = EncodingLayer(vector_size, hidden_size)
        self.inference = LocalInferenceModel()
        self.inferComp = InferenceComposition(hidden_size * 8, hidden_size, hidden_size, dropout)
        self.linear = LinearSoftmax(hidden_size * 8, hidden_size, num_labels, activation='tanh')

    def load_embeddings(self, vectors):#pre-trained vector
        self.embedding_layer.load(vectors)

    def forward(self, premise, hypothesis): #[batch, seq_len]
        # input embedding
        p_embeded = self.embedding_layer(premise)
        h_embeded = self.embedding_layer(hypothesis)

        p_ = self.encoder(p_embeded)
        h_ = self.encoder(h_embeded)

        # local inference
        p_mask, h_mask = (premise != 1).long(), (hypothesis != 1).long()
        m_p, m_h = self.inference(p_, h_, p_mask, h_mask)

        # inference composition
        v = self.inferComp(m_p, m_h, p_mask, h_mask)

        # final multi-layer perceptron
        logits = self.linear(v) #[batch, class_num]
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities
