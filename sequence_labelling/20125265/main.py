import torch
import argparse
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from sklearn.metrics import  classification_report

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:4')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--weight_decay', type=float, default=0.5)
parser.add_argument('--ckp', type=str, default='/data/yinli/task2/models/model3.pt')
args = parser.parse_args()
device = torch.device(args.cuda)

#加载数据集并处理
import os
from codecs import open

def load_data(root, file_name):
    """读取数据"""
    word_lists = []
    tag_lists = []
    with open(os.path.join(root, file_name), 'r',) as f:
        lines = [line.strip('\n').split(' ') for line in f.readlines()]
        word_list = []
        tag_list = []
        for line in lines:
            if line==['']:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                word_list = []
                tag_list = []
            else:
                word_list.append(line[0].lower())
                tag_list.append(line[-1])
        word_lists.append(word_list)
        tag_lists.append(tag_list)
    return word_lists, tag_lists

def load_test_data(root, file_name):
    with open(os.path.join(root, file_name)) as f:
        lines = [line.strip('\n').lower().split(' ') for line in f.readlines()]
    return lines

root = '/data/yinli/dataset/task2'
train_data, train_label = load_data(root, 'train.txt')
valid_data, valid_label = load_data(root, 'valid.txt')
pred_data = load_test_data(root, 'conll03.txt')

print("train_data_len",len(train_data))
print("valid_data_len",len(valid_data))
print("test2_data_len",len(pred_data))

from gensim.models import word2vec

#训练词向量
def train_word2vec(x):
    #训练word to vector 的 word embedding
    model = word2vec.Word2Vec(x, size=300, window=5, min_count=2, workers=12, iter=10, sg=1)
    return model

if __name__ == "__main__":
    model = train_word2vec(train_data)

    print("saving model ...")
    model.save('/data/yinli/task2/w2v_all.model')

from gensim.models import Word2Vec

class Preprocess():
    def __init__(self, w2v_path):
        self.w2v_path = w2v_path
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

w2v_path = '/data/yinli/task2/w2v_all.model'
#获得单词词典、标注词典和预训练词向量
preprocess = Preprocess(w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
word2idx = preprocess.word2idx
tag2idx =  {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, '<START>':9, '<STOP>':10}
print(tag2idx)

#将数据转换成id
train_x = [[word2idx.get(word, word2idx['<UNK>']) for word in words] for words in train_data]
train_y = [[tag2idx[tag] for tag in tags] for tags in train_label]
valid_x = [[word2idx.get(word, word2idx['<UNK>']) for word in words] for words in valid_data]
valid_y = [[tag2idx[tag] for tag in tags] for tags in valid_label]

pred_x = [[word2idx.get(word, word2idx['<UNK>']) for word in words] for words in pred_data]

#模型定义（参考梁棋棋同学）
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, embed_size, output_size, embedding, tag2id):
        super(BiLSTM_CRF, self).__init__()

        #----------------BiLSTM的参数————————————————
        self.embed_size = embed_size
        self.hidden_size = args.hidden_size
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.embedding.weight = nn.Parameter(embedding)
        self.embedding.weight.requires_grad = False
        self.bilstm = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=args.num_layers,\
                              batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2*self.hidden_size, output_size)
        )
        #-------------------CRF的参数------------------------
        self.tagset_size = len(tag2id)
        self.tag2id = tag2id
        #转移矩阵;transitions[i][j]表示标签 j 转移到标签 i 的概率
        self.transitions = torch.nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        #约束条件
        self.transitions.data[self.tag2id['<START>'], :] = -100000 #没有标签能转向<START>
        self.transitions.data[:, self.tag2id['<STOP>']] = -100000 #<STOP> 标签不能转向其他标签

    # -------------------------模型前向传播-------------------
    def forward(self, sentence, batch_lengths):
        lstm_feats = self.BiLSTM(sentence, batch_lengths)
        score, batch_tags = self.decode(lstm_feats)
        return score, batch_tags#[batch_size, seq_len]

    #--------------------BiLSTM----------------------------
    def BiLSTM(self, sentence, batch_lengths):
        batch_size = sentence.size(0)
        embed = self.embedding(sentence) #[batch_size, seq_len, emb_size]

        h = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(sentence.device)
        c = torch.zeros(2*args.num_layers, batch_size, self.hidden_size).to(sentence.device)

        input = pack_padded_sequence(embed, batch_lengths, batch_first=True)
        output, hidden = self.bilstm(input, (h, c))
        output, _ = pad_packed_sequence(output, batch_first=True) #output[batch_size, seq_len, hidden_size*2]
        lstm_feats = self.fc(output) #[batch_size, seq_len, output_size]
        return lstm_feats

    #----------------------CRF---------------------------
    def decode(self, feats):
        backpointers = []
        bptrs_t = []
        viterbivars_t = []

        forward_var = torch.full([feats.shape[0], self.tagset_size], -1000).to(device)  #[batch_size, out_size]
        forward_var[:, self.tag2id['<START>']] = 0
        for feat in range(feats.shape[1]):#[seq_len]
            next_tag_var = torch.stack([forward_var] * feats.shape[2]).transpose(0, 1) + self.transitions #[batch_size, out_size, out_size]
            viterbivars, best_tag_id = torch.max(next_tag_var, dim=2) #[batch_size, out_size]表示在当前的标签i（未定）的情况下，前一个最大可能的标签j
            viterbivars, best_tag_id = viterbivars.squeeze(0), best_tag_id.squeeze(0)
            bptrs_t.append(best_tag_id.detach().cpu().numpy().tolist())
            viterbivars_t.append(viterbivars.detach().cpu().numpy().tolist())
            forward_var = (viterbivars + feats[:, feat, :])
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag2id['<STOP>']]  #[batch_size, out_size]
        path_score, best_tag_id = torch.max(terminal_var, dim=1) #[batch_size] 末尾转向标签STOP概率最大的 tagid
        path_score, best_tag_id = path_score.squeeze(0).view(-1, 1), best_tag_id.squeeze(0).view(-1, 1).detach().cpu().numpy().tolist()

        #根据动态规划，由最后的节点，向前选取最佳的路径
        best_path = [best_tag_id] #[1, batch_size, 1]
        for bptrs in reversed(torch.Tensor(bptrs_t).long()): #[seq_len, batch_size, out_size]
            best_tag_id = [bptrs[i][best_tag_id[i]].numpy().tolist() for i in range(feats.shape[0])] #[batch_size, 1]
            best_path.append(best_tag_id)
        #best_path [seq_len, batch_size, 1]
        best_path = torch.Tensor(best_path).long().permute(2,0,1).squeeze(0).numpy().tolist() #[1， seq_len, batch_size]
        start = best_path.pop() #将<START>标签弹出
        best_path.reverse() #将序列倒回正序 [seq_len, batch_size]
        return path_score, torch.Tensor(best_path).long().transpose(1,0).numpy().tolist() #[batch_size, seq_len]

    def loss_function(self, sentence, tags, batch_lengths):
        feats = self.BiLSTM(sentence, batch_lengths)
        # loss = log(∑ e^s(X,y)) - s(X,y)
        forward_score = self._forward_alg(feats)# loss的log部分的结果
        gold_score = self._score_sentence(feats, tags)# loss的S(X,y)部分的结果
        return torch.sum(forward_score - gold_score)

    #--------------计算loss的log部分---------------------
    def _forward_alg(self, feats):
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -1000.).to(device) #[batch_size, out_size]
        init_alphas[:, self.tag2id['<START>']] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)

        for feat_index in range(feats.shape[1]):#[seq_len]
            forward_score = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1) #[batch_size, out_size, out_size]
            emit_score = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2) #[batch_size, out_size, 1]
            total_score = forward_score + emit_score + torch.unsqueeze(self.transitions, 0) #[batch_size, out_size, out_size]
            forward_var_list.append(torch.logsumexp(total_score, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag2id['<STOP>']].repeat([feats.shape[0], 1])#[batch_size, out_size]
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    #-----------------计算loss的S(X,y)部分------------------
    def _score_sentence(self, feats, tags):
        tags = torch.tensor(tags).to(device)
        score = torch.zeros(tags.shape[0]).to(device)
        start = torch.full([tags.shape[0],1], self.tag2id['<START>']).long().to(device)
        tags = torch.cat((start, tags), dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + self.transitions[tags[:, i+1], tags[:,i]] + feat[range(feat.shape[0]), tags[:,i+1]]
        score = score + self.transitions[self.tag2id['<STOP>'], tags[:,-1]]
        return score

# 将输入的数据按照句子的长度从大到小排列
def sort_by_lengths(word_lists, tag_lists=None):
    if tag_lists is not None:
        pairs = list(zip(word_lists, tag_lists))
        indices = sorted(range(len(pairs)),
                         key=lambda k: len(pairs[k][0]),
                         reverse=True)
        pairs = [pairs[i] for i in indices]

        word_lists, tag_lists = list(zip(*pairs))
        return word_lists, tag_lists, indices
    else:
        indices = sorted(range(len(word_lists)),
                         key=lambda k: len(word_lists[k]),
                         reverse=True)
        word_lists = [word_lists[i] for i in indices]
        return word_lists, indices


# 将每一批次的句子进行pad，并转换成tensor
def pad_sentence(batch, dicts):
    PAD = dicts.get('<PAD>')

    max_len = len(batch[0])
    # batch各个元素的长度
    lengths = [len(data) for data in batch]

    batch_tensor = []
    for i, data in enumerate(batch):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            data.append(PAD)
        batch_tensor.append(data)
    return torch.LongTensor(batch_tensor), lengths


# 将每一批次的标签进行pad，并转换成tensor
def pad_label(batch):
    max_len = len(batch[0])
    batch_tensor = []
    for i, data in enumerate(batch):
        PAD = data[-1]
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            data.append(PAD)
        batch_tensor.append(data)
    return torch.LongTensor(batch_tensor)

#移除标签O
def remove_O(golden_tags_list, pred_tags_list):
    length = len(golden_tags_list)
    O_tag_indices = [i for i in range(length)
                     if golden_tags_list[i] == 0]

    golden_tags_list = [tag for i, tag in enumerate(golden_tags_list)
                        if i not in O_tag_indices]

    pred_tags_list = [tag for i, tag in enumerate(pred_tags_list)
                         if i not in O_tag_indices]
    print("原总标记数为{}，移除了{}个O标记，占比{:.2f}%".format(
        length,
        len(O_tag_indices),
        len(O_tag_indices) / length * 100
    ))
    return golden_tags_list, pred_tags_list


target_names = ['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC']
f1 = 0
#++++++++++++++++++++++++++++++++模型验证++++++++++++++++++++++++
def validate(model, dev_word_lists, dev_tag_lists):
    model.eval()
    golden_tags_list = []
    pred_tags_list = []
    global f1
    with torch.no_grad():
        val_losses = 0.
        val_step = 0
        for ind in range(0, len(dev_word_lists), args.batch_size):
            val_step += 1
            # 准备batch数据
            batch_sents = dev_word_lists[ind:ind + args.batch_size]
            batch_tags = dev_tag_lists[ind:ind + args.batch_size]
            test_sents, lengths = pad_sentence(batch_sents, word2idx)
            test_sents = test_sents.to(device)
            true_tags = pad_label(batch_tags)
            true_tags = true_tags.to(device)

            # forward
            scores, batch_tagids = model(test_sents, lengths)
            # 计算损失
            loss = model.loss_function(test_sents, true_tags, lengths).to(device)
            val_losses += loss.item()

            for i, ids in enumerate(batch_tagids):
                tag_list = []
                for j in range(lengths[i]):
                    tag_list.append(ids[j])
                pred_tags_list.extend(tag_list)

            for i, ids in enumerate(true_tags):
                tag_list = []
                for j in range(lengths[i]):
                    tag_list.append(ids[j].item())
                golden_tags_list.extend(tag_list)
            golden_tags_list, pred_tags_list = remove_O(golden_tags_list, pred_tags_list)

        val_loss = val_losses / val_step
        report = classification_report(golden_tags_list, pred_tags_list, output_dict=True)
        if report['macro avg']['f1-score'] > f1:
            f1 = report['macro avg']['f1-score']
            torch.save(model.state_dict(), args.ckp)
            print("save model......")
            print(classification_report(golden_tags_list, pred_tags_list, target_names=target_names, digits=6))

    return val_loss

#++++++++++++++++++++++++++++++++模型训练+++++++++++++++++++++
input_size = len(word2idx)
output_size = len(tag2idx)
embed_size = 300
model = BiLSTM_CRF(input_size, embed_size, output_size, embedding, tag2idx).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# 对数据集按照长度进行排序
word_lists, tag_lists, _ = sort_by_lengths(train_x, train_y)
dev_word_lists, dev_tag_lists, _ = sort_by_lengths(valid_x, valid_y)
for epoch in range(args.num_epoch):
    model.train()
    train_step = 0
    losses = 0.

    for ind in range(0, len(word_lists), args.batch_size):
        batch_sents = word_lists[ind:ind + args.batch_size]
        batch_tags = tag_lists[ind:ind + args.batch_size]

        train_step += 1
        # 准备数据
        train_sents, lengths = pad_sentence(batch_sents, word2idx)
        train_sents = train_sents.to(device)
        true_tags = pad_label(batch_tags)
        true_tags = true_tags.to(device)

        # forward
        scores, batch_tagids = model(train_sents, lengths)

        # 计算损失 更新参数
        optimizer.zero_grad()
        loss = model.loss_function(train_sents, true_tags, lengths).to(device)
        loss.backward()
        optimizer.step()
        losses += loss.item()

    train_loss = losses / train_step
    # 每轮结束测试在验证集上的性能，保存最好的一个
    print("-----------------------------开始验证----------------------------------------")
    val_loss = validate(model, dev_word_lists, dev_tag_lists)
    print("Epoch {}, Train Loss:{:.4f} , Val Loss:{:.4f}, F1:{:.4f}".format(epoch + 1, train_loss, val_loss, f1))
    print("-----------------------------------------------")


#-----------------------------开始预测--------------------------------
print("----------------------开始预测------------------------------")
if os.path.exists(args.ckp):
    print("loading model......")
    model.load_state_dict(torch.load(args.cpk))
# 准备数据
word_lists, indices = sort_by_lengths(pred_x)
model.eval()
targets_tag = []
with torch.no_grad():
    for ind in range(0, len(word_lists), args.batch_size):
        # 准备batch数据
        batch_sents = word_lists[ind:ind + args.batch_size]
        test_sents, lengths = pad_sentence(batch_sents, word2idx)
        test_sents = test_sents.to(device)

        # forward
        scores, batch_tagids = model(test_sents, lengths)

        for i, ids in enumerate(batch_tagids):
            tag_list = []
            for j in range(lengths[i]):# 真实长度
                tag_list.append(ids[j])
            targets_tag.append(tag_list)

# 将id转化为标注
pred_tag_lists = []
id2tag = dict((idx, tag) for tag, idx in tag2idx.items())
for i, tagids in enumerate(targets_tag):
    tag_list = []
    for id in tagids:
        tag_list.append(id2tag[id])
    pred_tag_lists.append(tag_list)

# 下面根据indices将pred_tag_lists转化为原来的顺序
ind_maps = sorted(list(enumerate(indices)), key=lambda e: e[1])
indices, _ = list(zip(*ind_maps))
pred_tag_lists = [pred_tag_lists[i] for i in indices]


#将预测结果记录下来
root = '/data/yinli/task2'
save_file = 'result.txt'
with open(os.path.join(root, save_file), 'w+') as f:
    for tags in pred_tag_lists:
        for tag in tags:
            f.write(tag + ' ')
        f.write('\n')