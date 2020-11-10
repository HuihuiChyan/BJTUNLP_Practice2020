import torch
import argparse
import data_proc
import train

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:6')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--ckp', type=str, default='ckp/model_3.pt')
parser.add_argument('--max_acc', type=float, default=0.5)
args = parser.parse_args()

if torch.cuda.is_available():
    print("using cuda......")
    device = torch.device(args.cuda)

# ======================================= 加载数据集并处理 ======================================= #

#读入训练集
with open('./snli_1.0/train_sentence1_split.txt', 'r', encoding='utf-8') as ftrain_feature1:
    train_feature1_line = [line.strip() for line in ftrain_feature1.readlines()]
with open('./snli_1.0/train_sentence2_split.txt', 'r', encoding='utf-8') as ftrain_feature2:
    train_feature2_line = [line.strip() for line in ftrain_feature2.readlines()]
with open('./snli_1.0/train_gold_label.txt', 'r', encoding='utf-8') as ftrain_label:
    train_label_line = [line.strip() for line in ftrain_label.readlines()]
#读入验证集
with open('./snli_1.0/dev_sentence1_split.txt', 'r', encoding='utf-8') as fdev_feature1:
    dev_feature1_line = [line.strip() for line in fdev_feature1.readlines()]
with open('./snli_1.0/dev_sentence2_split.txt', 'r', encoding='utf-8') as fdev_feature2:
    dev_feature2_line = [line.strip() for line in fdev_feature2.readlines()]
with open('./snli_1.0/dev_gold_label.txt', 'r', encoding='utf-8') as fdev_label:
    dev_label_line = [line.strip() for line in fdev_label.readlines()]
#读入测试集
with open('./snli_1.0/test_sentence1_split.txt', 'r', encoding='utf-8') as ftest_feature1:
    test_feature1_line = [line.strip() for line in ftest_feature1.readlines()]
with open('./snli_1.0/test_sentence2_split.txt', 'r', encoding='utf-8') as ftest_feature2:
    test_feature2_line = [line.strip() for line in ftest_feature2.readlines()]
with open('./snli_1.0/test_gold_label.txt', 'r', encoding='utf-8') as ftest_label:
    test_label_line = [line.strip() for line in ftest_label.readlines()]

#用split分隔开存入列表
train_feature1_line = [line.split(" ") for line in train_feature1_line]
train_feature2_line = [line.split(" ") for line in train_feature2_line]
dev_feature1_line = [line.split(" ") for line in dev_feature1_line]
dev_feature2_line = [line.split(" ") for line in dev_feature2_line]
test_feature1_line = [line.split(" ") for line in test_feature1_line]
test_feature2_line = [line.split(" ") for line in test_feature2_line]

#获得单词字典
#model_word2vec = Word2Vec(train_feature1_line+train_feature2_line, sg=1, min_count=1, size=128, window=5)
#model_word2vec.save('word2vec_model.txt')
w2v_model = KeyedVectors.load_word2vec_format('w2v_model/word2vec_model_1.txt',binary=False, encoding='utf-8')
print('loading word2vec_model......')
word2id = dict(zip(w2v_model.wv.index2word,range(len(w2v_model.wv.index2word))))                              # word -> id
id2word = {idx:word for idx,word in enumerate(w2v_model.wv.index2word)}                                       # id -> word
feature_pad = 0    #PAD:填充词
label2id = {'neutral':0, 'entailment':1, 'contradiction':2, '-':3}
label_pad = 0

#获得数据和标签序列
train_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in train_feature1_line]
train_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in train_feature2_line]
train_label = [[label2id[word] if word in label2id else label_pad] for word in train_label_line]
dev_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in dev_feature1_line]
dev_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in dev_feature2_line]
dev_label = [[label2id[word] if word in label2id else label_pad] for word in dev_label_line]
test_feature1 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature1_line]
test_feature2 = [[word2id[word] if word in word2id else feature_pad for word in line] for line in test_feature2_line]
test_label = [[label2id[word] if word in label2id else label_pad] for word in test_label_line]

sentence1_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
sentence2_field = Field(sequential=True, use_vocab = False, batch_first=True, fix_length = 50, pad_token = feature_pad)
label_field = Field(sequential=False, use_vocab=False)
fields = [('sentence1', sentence1_field), ('sentence2', sentence2_field), ('label', label_field)]

#获得训练集的Iterator
train_examples = []
for index in range(len(train_label)):
    train_examples.append(Example.fromlist([train_feature1[index], train_feature2[index], train_label[index]], fields))
train_set = Dataset(train_examples, fields)
train_iter = BucketIterator(train_set, batch_size=batch_size, device=device, shuffle=True)

#获得验证集的Iterator
dev_examples = []
for index in range(len(dev_label)):
    dev_examples.append(Example.fromlist([dev_feature1[index], dev_feature2[index], dev_label[index]], fields))
dev_set = Dataset(dev_examples, fields)
dev_iter = Iterator(dev_set, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

#获得测试集的Iterator
test_examples = []
for index in range(len(test_label)):
    test_examples.append(Example.fromlist([test_feature1[index], test_feature2[index], test_label[index]], fields))
test_set = Dataset(test_examples, fields)
test_iter = Iterator(test_set, batch_size=batch_size, device=device, train=False, shuffle=False, sort=False)

class ESIM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_vector):
        super(ESIM,self).__init__()
        
        self.hidden_size = hidden_size
        self.embedding = torch.nn.Embedding(input_size, hidden_size)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_vector))
        self.embedding.weight.requires_grad = False
        self.bilstm1 = torch.nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.bilstm2 = torch.nn.LSTM(input_size=hidden_size * 8, hidden_size=hidden_size, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 8, 2),
            nn.Dropout(0.5),
            nn.Linear(2, output_size),
            nn.Softmax(dim=-1)
        )
    
    def attention(self, seq1, seq2, mask1, mask2):
        # 首先计算出eij也就是相似度
        eik = torch.matmul(seq2, seq1.transpose(1, 2))
        ekj = torch.matmul(seq1, seq2.transpose(1, 2))
        
        # mask操作：将相似度矩阵中值为1（.的填充id）的那些值全部用-1e9mask掉
        eik = eik.masked_fill(mask1.unsqueeze(-1) == 1, -1e9)
        ekj = ekj.masked_fill(mask2.unsqueeze(-1) == 1, -1e9)        
        
        # 归一化用于后续加权计算
        eik = F.softmax(eik, dim=-1)
        ekj = F.softmax(ekj, dim=-1)
        
        # 通过相似度和b的加权和计算出ai，通过相似度和a的加权和计算出bj
        ai = torch.matmul(ekj, seq2)
        bj = torch.matmul(eik, seq1)
        return ai, bj
    
    def submul(self, x1, x2):
        # 计算差和积
        sub = x1 - x2
        mul = x1 * x2
        return torch.cat([sub, mul], -1)
    
    def pooling(self, x):
        # 通过平均池和最大池获得固定长度的向量，并拼接送至最终分类器
        p1 = F.avg_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1,2), x.size(1)).squeeze(-1)
        return torch.cat([p1,p2], 1)

    def forward(self, seq1, seq2, mask1, mask2):
        
        # ==================== embedding ==================== #
        seq1 = self.embedding(seq1)
        seq2 = self.embedding(seq2)

        # ==================== bilstm  ==================== #
        bi1_1, _ = self.bilstm1(seq1)
        bi1_2, _ = self.bilstm1(seq2)

        # ==================== attention ==================== #
        ai, bj = self.attention(bi1_1, bi1_2, mask1, mask2)
        # 计算差和积然后和原向量合并，对应论文中 ma=[-a;~a;-a-~a;-a*~a] 和 mb=[-b;~b;-b-~b;-b*~b]
        ma = torch.cat([bi1_1, ai, self.submul(bi1_1, ai)], -1)
        mb = torch.cat([bi1_2, bj, self.submul(bi1_2, bj)], -1)

        # ==================== bilstm ==================== #
        bi2_1, _ = self.bilstm2(ma)
        bi2_2, _ = self.bilstm2(mb)

        # ==================== fc ==================== #
        output_1 = self.pooling(bi2_1)
        output_2 = self.pooling(bi2_2)
        output = torch.cat([output_1, output_2], -1)
        output = self.fc(output)
        
        return output

def dev_evaluate(device, net, dev_iter, max_acc, ckp):

    dev_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    loss_func = torch.nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for batch in dev_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))

            loss = loss_func(out, label.squeeze(-1))

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            #测试集评价指标
            out_epoch.extend(prediction)
            label_epoch.extend(label)
            dev_l += loss.item()
            n += 1

        acc = accuracy_score(label_epoch, out_epoch)
        if acc > max_acc : 
            max_acc = acc
            torch.save(net.state_dict(), ckp)
            print("save model......")

    return dev_l/n, acc, max_acc

def test_evaluate(device, net, test_iter):

    test_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    loss_func = torch.nn.CrossEntropyLoss()
    net.eval()
    with torch.no_grad():
        for batch in test_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))

            loss = loss_func(out, label.squeeze(-1))

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            #测试集评价指标
            out_epoch.extend(prediction)
            label_epoch.extend(label)
            test_l += loss.item()
            n += 1

        acc = accuracy_score(label_epoch, out_epoch)

    return test_l/n, acc

def training(device, w2v_model, train_iter, dev_iter, test_iter, batch_size, num_epoch, lr, weight_decay, ckp, max_acc):

    embedding_matrix = w2v_model.wv.vectors
    input_size, hidden_size = embedding_matrix.shape[0], embedding_matrix.shape[1]
    loss_func = torch.nn.CrossEntropyLoss()
    net = model.ESIM(input_size, hidden_size, 4, embedding_matrix).to(device)
    #net.load_state_dict(torch.load(ckp, map_location='cpu'))
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(num_epoch):
        net.train()
        train_l, n = 0.0, 0
        start = datetime.datetime.now()
        out_epoch, label_epoch = [], []
        for batch in train_iter:
            
            seq1 = batch.sentence1
            seq2 = batch.sentence2
            label = batch.label
            mask1 = (seq1 == 1)
            mask2 = (seq2 == 1)
            out = net(seq1.to(device),seq2.to(device), mask1.to(device), mask2.to(device))
            
            loss = loss_func(out, label.squeeze(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()

            out_epoch.extend(prediction)
            label_epoch.extend(label)

            train_l += loss.item()
            n += 1

        train_acc = accuracy_score(label_epoch, out_epoch)

        dev_loss, dev_acc, max_acc = dev_evaluate(device, net, dev_iter, max_acc, ckp)
        test_loss, test_acc = test_evaluate(device, net, test_iter)
        end = datetime.datetime.now()

        print('epoch %d, train_acc %f, dev_acc %f, test_acc %f, max_acc %f, time %s'% (epoch+1, train_acc, dev_acc, test_acc, max_acc, end-start))

training(device, w2v_model, train_iter, dev_iter, test_iter, args.batch_size, args.num_epoch, args.lr, args.weight_decay, args.ckp, args.max_acc)