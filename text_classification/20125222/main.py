import re
from itertools import chain
import gensim
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import time
from sklearn.metrics import accuracy_score

# 返回数组，分别为文本+特征
def read_files(path, filetype):
    file_list = []
    pos_path = path + filetype + "/pos/"
    neg_path = path + filetype + "/neg/"
    for f in os.listdir(pos_path):
        file_list += [[pos_path + f, 1]]
    for f in os.listdir(neg_path):
        file_list += [[neg_path + f, 0]]
    data = []
    for fi, label in file_list:
        with open(fi, encoding='utf8') as fi:
            data += [[" ".join(fi.readlines()), label]]
    return data

# 去掉停用词+还原词性
# def stop_words(text):
#     stop_words = stopwords.words("english")
#     wl = WordNetLemmatizer()
#     words = wl.lemmatize(text)
#     return words

# 缩略词还原：不过处理贼慢...
replacement_patterns = [
    (r'won\'t', 'will not'),
    (r'can\'t', 'cannot'),
    (r'i\'m', 'i am'),
    (r'ain\'t', 'is not'),
    (r'(\w+)\'ll', '\g<1> will'),
    (r'(\w+)n\'t', '\g<1> not'),
    (r'(\w+)\'ve', '\g<1> have'),
    (r'(\w+)\'s', '\g<1> is'),
    (r'(\w+)\'re', '\g<1> are'),
    (r'(\w+)\'d', '\g<1> would')]


class RegexpReplacer(object):
    def __init__(self, patterns=replacement_patterns):
        self.patterns = [(re.compile(regex), repl) for (regex, repl) in patterns]

    def replace(self, text):
        s = text
        for (pattern, repl) in self.patterns:
            (s, count) = re.subn(pattern, repl, s)
        return s

replacer = RegexpReplacer()

# 特殊处理数据+去标点
def data_process(text):
    text = text.lower()
    # 特殊数据处理，该地方参考的殷同学的
    text = text.replace("<br /><br />", "").replace("it's", "it is").replace("i'm", "i am").replace("he's",
                                                                                                    "he is").replace(
        "she's", "she is") \
        .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's",
                                                                                                        "that is") \
        .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't",
                                                                                                     "does not") \
        .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not") \
        .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's",
                                                                                                          "let us") \
        .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't",
                                                                                                        "could not") \
        .replace("you'll", "you will").replace("you've", "you have")
    # 去除标点
    text = re.sub("[^a-zA-Z']", "", text.lower())
    text = " ".join([word for word in text.split(' ')])
    return text

def get_token_text(text):
    token_data = [data_process(st) for st in text.split()]
    # token_data = [st.lower() for st in text.split()]
    token_data = list(filter(None, token_data))
    return token_data


def get_token_data(data):
    data_token = []
    for st, label in data:
        data_token.append(get_token_text(st))
    return data_token


# 建立词典
def get_vocab(data):
    vocab = set(chain(*data))
    vocab_size = len(vocab)
    word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
    word_to_idx['<unk>'] = 0
    idx_to_word = {i + 1: word for i, word in enumerate(vocab)}
    idx_to_word[0] = '<unk>'
    return vocab, vocab_size, word_to_idx, idx_to_word

# 转化为索引
def encode_st(token_data, vocab, word_to_idx):
    features = []
    for sample in token_data:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features

# 填充和截断
def pad_st(features, maxlen, pad=0):
    padded_features = []
    for feature in features:
        if len(feature) > maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while (len(padded_feature) < maxlen):
                padded_feature.append(pad)
        padded_features.append(padded_feature)
    return padded_features

# 处理验证集的
def read_file(data_path):
    data = []
    for line in open(data_path):
        token_data = [data_process(st) for st in line.split()]
        token_data = list(filter(None, token_data))
        data.append(token_data)
    return data

data_path = "data/aclImdb/"
save_path = ""
maxlen = 300

train_data = read_files(data_path, "train")
test_data = read_files(data_path, "test")
vail_data = read_file('test.txt')

print("read_file success!")

train_token = get_token_data(train_data)
test_token = get_token_data(test_data)

print("get_token_data success!")

vocab, vocab_size, word_to_idx, idx_to_word = get_vocab(train_token)
# np.save("vocab.npy",vocab)

print("vocab_save success!")
vail_data = pad_st(encode_st(vail_data, vocab, word_to_idx), maxlen)
train_features = pad_st(encode_st(train_token, vocab, word_to_idx), maxlen)
test_features = pad_st(encode_st(test_token, vocab, word_to_idx), maxlen)

train_label = [score for _, score in train_data]
test_label = [score for _, score in test_data]

# 参考博文https://cloud.tencent.com/developer/article/1491567、https://wenjie.blog.csdn.net/article/details/108436234
class textCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, seq_len, labels, weight, droput, **kwargs):
        super(textCNN, self).__init__(**kwargs)
        self.embedding_S = nn.Embedding(vocab_size, embed_size)
        self.embedding_S.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_S.weight.requires_grad = False
        self.embedding_D = nn.Embedding(vocab_size, embed_size)
        self.embedding_D.weight.data.copy_(weight)
        # 是否将embedding定住
        self.embedding_D.weight.requires_grad = True
        num_filters = 256
        self.labels = labels
        self.conv1 = nn.Conv2d(1,num_filters,(2,embed_size))
        self.conv2 = nn.Conv2d(1,num_filters,(3,embed_size))
        self.conv3 = nn.Conv2d(1,num_filters,(4,embed_size))
        self.conv4 = nn.Conv2d(1,num_filters,(5,embed_size))
        self.pool1 = nn.MaxPool2d((seq_len - 2 + 1, 1))
        self.pool2 = nn.MaxPool2d((seq_len - 3 + 1, 1))
        self.pool3 = nn.MaxPool2d((seq_len - 4 + 1, 1))
        self.pool4 = nn.MaxPool2d((seq_len - 5 + 1, 1))
        self.dropout = nn.Dropout(droput)
        self.linear = nn.Linear(2*num_filters*4,labels)
    def forward(self, x):
        #两层通道
        out1 = self.embedding_S(x).view(x.shape[0],1,x.shape[1],-1)
        out2 = self.embedding_D(x).view(x.shape[0],1,x.shape[1],-1)
        #对第一层进行卷积
        x1 = F.relu(self.conv1(out1))
        x2 = F.relu(self.conv2(out1))
        x3 = F.relu(self.conv3(out1))
        x4 = F.relu(self.conv4(out1))
        #对第一层进行池化
        x1 = self.pool1(x1)
        x2 = self.pool2(x2)
        x3 = self.pool3(x3)
        x4 = self.pool4(x4)
        #对第二层进行卷积
        x5 = F.relu(self.conv1(out2))
        x6 = F.relu(self.conv2(out2))
        x7 = F.relu(self.conv3(out2))
        x8 = F.relu(self.conv4(out2))
        #对第二层进行池化
        x5 = self.pool1(x5)
        x6 = self.pool2(x6)
        x7 = self.pool3(x7)
        x8 = self.pool4(x8)
        #合体～
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8),1)
        x = x.view(x.shape[0], 1, -1)
        x = self.dropout(x)
        x = self.linear(x)
        x = x.view(-1, self.labels)
        return x


embed_size = 300
num_hidens = 100
num_layers = 2
bidirectional = True
batch_size = 256
labels = 2
lr = 0.0001
use_gpu = True
seq_len = 300
droput = 0.5
num_epochs = 100

best_test_acc = 0

word2vec_path = "data/glove_to_word2vec.txt"
output_path = "r2.txt"

train_features = torch.tensor(train_features)
test_features = torch.tensor(test_features)
train_labels = torch.tensor(train_label)
test_labels = torch.tensor(test_label)
vail_features = torch.tensor(vail_data)
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, encoding='utf-8')
weight = torch.zeros(vocab_size + 1, embed_size)
for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        print("失败")
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))
#运行十次寻找高评分结果
for i in range(10):
    net = textCNN(vocab_size=(vocab_size + 1), embed_size=embed_size, seq_len=seq_len, labels=labels, weight=weight,
                  droput=droput)
    os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
    net = nn.DataParallel(net)
    net = net.cuda()
    train_set = torch.utils.data.TensorDataset(train_features, train_labels)
    test_set = torch.utils.data.TensorDataset(test_features, test_labels)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    vail_set = torch.utils.data.TensorDataset(vail_features, test_labels)
    vail_iter = torch.utils.data.DataLoader(vail_set, batch_size=batch_size, shuffle=False)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(filter(lambda p:p.requires_grad, net.parameters()), lr = lr)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
    for epoch in range(num_epochs):
        f = open(output_path, "a")
        start = time.time()
        train_loss, test_losses = 0, 0
        train_acc, test_acc = 0, 0
        n, m = 0, 0
        net.train()
        for feature, label in train_iter:
            n += 1
            feature = feature.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(score.cpu().data, dim=1), label.cpu())
            train_loss += loss
        with torch.no_grad():
            net.eval()
            for test_feature, test_label in test_iter:
                m += 1
                test_feature = test_feature.cuda()
                test_label = test_label.cuda()
                test_score = net(test_feature)
                test_loss = loss_function(test_score, test_label)
                test_acc += accuracy_score(torch.argmax(test_score.cpu().data, dim=1), test_label.cpu())
                test_losses += test_loss
        end = time.time()
        runtime = end - start
        f = open(output_path, "a")
        f.write(
            'epoch: %d, train loss: %.4f, train acc: %.5f, test loss: %.4f, test acc: %.5f, best test acc: %.5f,time: %.4f \n' % (
                epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, best_test_acc / m,
                runtime))
        f.close()
        #有高评分时保存+预测
        if (test_acc > best_test_acc and test_acc / m > 0.88):
            torch.save(net, 'best.pth')
            print("test_acc / m > 0.88")
            f = open("end.txt", "a")
            net.eval()
            for x, y in vail_iter:
                x = x.cuda()
                test_score = net(x)
                print(torch.argmax(test_score.cpu().data, dim=1))
                f.write(str(torch.argmax(test_score.cpu().data, dim=1)))
            f.close()
            break
