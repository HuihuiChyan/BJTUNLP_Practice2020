import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

# 保证每次结果一样
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

#加载数据
def load_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip('\n').split(' ') for line in lines]
    x = [data[1:] for data in datas]
    y = [data[0] for data in datas]
    return x, y

#加载数据
root = '/data/yinli/dataset/task1/'
train_data, train_label = load_data(os.path.join(root, 'train.txt'))
dev_data, dev_label = load_data(os.path.join(root, 'dev.txt'))

import gensim

class Preprocess():
    def __init__(self, sen_len, w2v_path, min_freq=1):
        self.w2v_path = w2v_path
        self.sen_len = sen_len
        self.min_freq = min_freq
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    # 下载预训练模型
    def get_w2v_model(self):
        self.embedding = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
        self.embedding_dim = self.embedding.vector_size

    def build_vocab(self, sentences):
        vocab_dic = {}
        for words in sentences:
            for word in words:
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= self.min_freq], key=lambda x: x[1], reverse=True)
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        vocab_dic.update({'<PAD>': len(vocab_dic), '<UNK>': len(vocab_dic) + 1})
        return vocab_dic

    def make_embedding(self, sentences, load=True):
        print('Get embedding...')
        if load:
            print('loading word to vec model...')
            self.get_w2v_model()
        else:
            raise NotImplementedError

        self.word2idx = self.build_vocab(sentences)
        # 制作 word2idx、idx2word、word2vector
        for item in self.word2idx.items():
            self.idx2word.append(item[0])
            if item[0] in self.embedding:
                self.embedding_matrix.append(self.embedding[item[0]])
            else:
                vector = np.random.uniform(0, 1, (self.embedding_dim, 1))
                self.embedding_matrix.append(vector)

        self.embedding_matrix = torch.tensor(self.embedding_matrix, dtype=torch.float)
        print('total words:{}'.format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sentence(self, sentence):
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx['<PAD>'])
        assert len(sentence) == sen_len
        return sentence

    def sentence_word2id(self, sentences):
        sentence_list = []
        for i, sen in enumerate(sentences):
            sentence_idx = []
            for word in sen:
                if word in self.word2idx.keys():
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx['<UNK>'])
            # 填充句子，使其为特定长度
            sentence_idx = self.pad_sentence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def label_to_tensor(self, labels):
        y = []
        for label in labels:
            if label == 'neg':
                y.append(0)
            else:
                y.append(1)
        return torch.LongTensor(y)

class MyDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]
        else:
            return self.data[idx], self.label[idx]


# 模型
class Model(nn.Module):
    def __init__(self, embedding, num_filters, filter_sizes, dropout=0.5, fix_embedding=True):
        super(Model, self).__init__()
        # 制作embedding layer
        self.embedding = nn.Embedding(embedding.size(0), embedding.size(1))
        self.embedding.weight = nn.Parameter(embedding)
        # 是否将embedding定住
        self.embedding.weight.requires_grad = False if fix_embedding else True
        self.embedding_dim = embedding.size(1)
        self.dropout = dropout
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, num_filters, (k, self.embedding_dim)) for k in filter_sizes]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(num_filters * len(filter_sizes), 2),
            nn.Softmax(dim=1)
        )

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.classifier(out)
        return out

def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    model.train()
    criterion = nn.CrossEntropyLoss() #定义损失函数
    t_batch = len(train)
    v_batch = len(valid)
    best_acc = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(n_epoch):
        print("Epoch:", epoch+1)
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(train):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            prediction = outputs.argmax(dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            correct = (prediction == labels).sum().item()
            total_acc += correct
            total_loss += loss.item()
        print('Train | Loss:{:.5f} Acc: {:.4f}'.format(total_loss / t_batch, total_acc / (t_batch * batch_size)))

        #这段做validation
        total_acc, total_loss = 0, 0
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(valid):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                prediction = outputs.argmax(dim=1)
                loss = criterion(outputs, labels)
                correct = (prediction == labels).sum().item()
                total_acc += correct
                total_loss += loss.item()
            print("Valid | Loss:{:.5f} Acc: {:.4f} ".format(total_loss / v_batch, total_acc / (v_batch * batch_size)))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save({'state_dict': model.state_dict()}, "{}/{}_ckpt.model".format(model_dir,model_name))

        print("-----------------------------------------------------\n")
        model.train()

w2v_path = '/data/yinli/dataset/task1/GoogleNews-vectors-negative300.bin'
model_dir = '/data/yinli/dataset/task1'
model_name = 'model_word2vec_Cross'
sen_len = 300
fix_embedding = True

preprocess = Preprocess(sen_len, w2v_path=w2v_path, min_freq=2)
embeddig = preprocess.make_embedding(train_data, load=True)

train_x = preprocess.sentence_word2id(train_data)
train_y = preprocess.label_to_tensor(train_label)

valid_x = preprocess.sentence_word2id(dev_data)
valid_y = preprocess.label_to_tensor(dev_label)

num_filters = 512
filter_sizes = [2, 3, 4]
batch_size = 100
lr = 0.001

train_dataset = MyDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataset = MyDataset(valid_x, valid_y)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
model = Model(embeddig, num_filters, filter_sizes, dropout=0.6, fix_embedding=fix_embedding)
model = model.to(device)
#开始训练
n_epoch = 20
training(batch_size, n_epoch, lr, model_dir, train_loader, valid_loader, model, device)

#加载模型
model = Model(embeddig, num_filters, filter_sizes, dropout=0.5, fix_embedding=fix_embedding)
checkpoint = torch.load(os.path.join(model_dir, model_name+"_ckpt.model"))
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)


def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip('\n').split(' ') for line in lines]
    return datas


def testing(test_loader, model, device):
    model.eval()
    ret_output = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            prediction = outputs.argmax(dim=1)
            ret_output += prediction.tolist()

    with open(os.path.join(root, 'result4_2_888.txt'), 'w+') as f:
        for label in ret_output:
            tag = 'neg' if label == 0 else 'pos'
            f.write(tag + '\n')
    return ret_output

#加载测试数据
root = '/data/yinli/dataset/task1/'
test_data = load_test_data(os.path.join(root, 'test.txt'))
test_x = preprocess.sentence_word2id(test_data)
test_dataset = MyDataset(x=test_x, y=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#开始预测
result = testing(test_loader, model, device)