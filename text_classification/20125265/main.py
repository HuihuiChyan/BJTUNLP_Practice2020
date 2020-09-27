import torch
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
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

#数据处理

root = 'E:\\jupyter_notebook\\task1\\data\\aclImdb\\'
class_name = ['neg', 'pos']

def replace_abbreviations(text):
    text = text.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's", "she is")\
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's", "that is")\
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't", "does not")\
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not")\
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's", "let us")\
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't", "could not")\
            .replace("you'll", "you will").replace("you've", "you have")
    return text

def clear_review(text):
    text = text.replace("<br /><br />", "")
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    return text

def stemed_words(text):
    stop_words = stopwords.words("english")
    lemma = WordNetLemmatizer()
    #去停用词，并将动词变回原形
    words = [lemma.lemmatize(w, pos='v') for w in text.split() if w not in stop_words]
    result = " ".join(words)
    return result

def process(text):
    text = replace_abbreviations(text)
    text = clear_review(text)
    text = stemed_words(text)
    return text

#处理训练数据
train_path = 'train.txt'
file = open(os.path.join(root, train_path), 'w+', encoding='utf-8')
for name in class_name:
    file_name_list = os.listdir(os.path.join(root, 'train', name))
    for f_name in file_name_list:
        with open(os.path.join(root, 'train', name, f_name), encoding='utf-8') as f:
            line = f.read()
            line = process(line)
            file.write(name + ' ' + line + '\n')
file.close()

#处理验证数据
dev_path = 'dev.txt'
file = open(os.path.join(root, dev_path), 'w+', encoding='utf-8')
for name in class_name:
    file_name_list = os.listdir(os.path.join(root, 'test', name))
    for f_name in file_name_list:
        with open(os.path.join(root, 'test', name, f_name), encoding='utf-8') as f:
            line = f.read()
            line = process(line)
            file.write(name + ' ' + line + '\n')
file.close()



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

#预训练词向量模型
from gensim.models import word2vec
from gensim.models import Word2Vec

def train_word2vec(x):
    #训练word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=300, window=5, min_count=5, workers=12, iter=10, sg=1)
    return model


model = train_word2vec(train_data)
print("saving model ...")
model.save('/data/yinli/dataset/task1/w2v_all_min.model')


#数据处理类
class Preprocess():
    def __init__(self, sen_len, w2v_path):
        self.w2v_path = w2v_path
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    # 下载预训练模型
    def get_w2v_model(self):
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print('Get embedding...')
        if load:
            print('loading word to vec model...')
            self.get_w2v_model()
        else:
            raise NotImplementedError

        # 制作 word2idx、idx2word、word2vector
        for i, word in enumerate(self.embedding.wv.vocab):
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])

        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        self.add_embedding('<PAD>')
        self.add_embedding('<UNK>')
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

#数据集结构
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

#训练
def training(batch_size, n_epoch, lr, model_dir, train, valid, model, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
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
        print('Train | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / t_batch, total_acc / (t_batch * batch_size)))

        #validation
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
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / v_batch, total_acc / (v_batch * batch_size)))
            if total_acc > best_acc:
                best_acc = total_acc
                torch.save({'state_dict': model.state_dict()}, "{}/{}_ckpt.model".format(model_dir,model_name))

        print("-----------------------------------------------------\n")
        model.train()


w2v_path = '/data/yinli/dataset/task1/w2v_all_min.model'
model_dir = '/data/yinli/dataset/task1'
model_name = 'model2'
sen_len = 150

preprocess = Preprocess(sen_len, w2v_path=w2v_path)
embeddig = preprocess.make_embedding(load=True)

train_x = preprocess.sentence_word2id(train_data)
train_y = preprocess.label_to_tensor(train_label)

valid_x = preprocess.sentence_word2id(dev_data)
valid_y = preprocess.label_to_tensor(dev_label)

num_filters = 256
filter_sizes = [2, 3, 4]
batch_size = 50
fix_embedding = True
lr = 0.001

train_dataset = MyDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
valid_dataset = MyDataset(valid_x, valid_y)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(embeddig, num_filters, filter_sizes, dropout=0.5, fix_embedding=fix_embedding)
model = model.to(device)

n_epoch = 20
#开始训练
training(batch_size, n_epoch, lr, model_dir, train_loader, valid_loader, model, device)



#测试
#加载模型
model = Model(embeddig, num_filters, filter_sizes, dropout=0.5, fix_embedding=fix_embedding)
checkpoint = torch.load(os.path.join(model_dir, model_name+"_ckpt.model"))
model.load_state_dict(checkpoint['state_dict'])
model = model.to(device)

#处理测试数据
test_path = 'test.txt'
file = open(os.path.join(root, test_path), 'w+', encoding='utf-8')
with open(os.path.join(root, 'test_raw.txt'), encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = process(line)
        file.write(line + '\n')
    print(lines[0])
file.close()

def load_test_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        datas = [line.strip('\n').split(' ') for line in lines]
    return datas

test_data = load_test_data(os.path.join(root, 'test.txt'))
test_x = preprocess.sentence_word2id(test_data)
test_dataset = MyDataset(x=test_x, y=None)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#测试函数
def testing(test_loader, model, device):
    model.eval()
    result = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device, dtype=torch.long)
            outputs = model(inputs)
            prediction = outputs.argmax(dim=1)
            result += prediction.int().tolist()
    with open(os.path.join(root, 'result2.txt'), 'w+', encoding='UTF-8') as f:
        for y in result:
            label = 'neg' if y == 0 else 'pos'
            f.write(label + '\n')
    return result

#开始测试
result = testing(test_loader, model, device)