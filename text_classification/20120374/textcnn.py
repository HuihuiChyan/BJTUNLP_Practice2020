import torch
import torch.nn as nn
import torch.nn.functional as F


class text_cnn(nn.Module):
    def __init__(self, vocab_size, weight_matrix,embedding_dim, n_filters, filters_sizes, sentence_max_len, output_dim, dropout):
        """
        :param vec_dim: 词向量维度
        :param n_filters: 每种卷积核个数
        :param filters_sizes: 卷积核大小
        :param output_dim: 输出维度
        :param dropout: dropout
        :param sentence_max_len:句子最大维度
        """
        super(text_cnn, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.copy_(weight_matrix)
        self.embedding.weight.requires_grad = False

        self.conv_0 = nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filters_sizes[0], embedding_dim))
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filters_sizes[1], embedding_dim))
        self.conv_2 = nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filters_sizes[2], embedding_dim))
        self.conv_3 = nn.Conv2d(in_channels=1, out_channels=n_filters,
                                kernel_size=(filters_sizes[3], embedding_dim))
        self.fc = nn.Linear(len(filters_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        x = self.embedding(text)
        #(batch,len,embed_dim)
        x = x.unsqueeze(1)
        # (batch,1,len,embed_dim)

        conved_0 = F.relu(self.conv_0(x).squeeze(3))
        conved_1 = F.relu(self.conv_1(x).squeeze(3))
        conved_2 = F.relu(self.conv_2(x).squeeze(3))
        conved_3 = F.relu(self.conv_3(x).squeeze(3))
        
        #(batch,n_filters,len-filter_size[n]+1

        pooled_0 = F.max_pool1d(conved_0, conved_0.shape[2]).squeeze(2)
        pooled_1 = F.max_pool1d(conved_1, conved_1.shape[2]).squeeze(2)
        pooled_2 = F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        pooled_3 = F.max_pool1d(conved_3, conved_3.shape[2]).squeeze(2)
        
        #(batch,n_filters)

        cat = self.dropout(torch.cat((pooled_0, pooled_1, pooled_2, pooled_3), dim=1))

        return self.fc(cat)
