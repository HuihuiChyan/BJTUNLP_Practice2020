import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.utils.data as Data
import torch.optim as opt
import pdb
from collections import Counter, defaultdict
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import nltk
import argparse
from sacremoses import MosesTokenizer
import os
import pickle

# 定义parser来存放参数
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--embedding_dim', type=int, default=200)
parser.add_argument('--sentence_len', type=int, default=300)
parser.add_argument('--window_size', type=list, default=[3, 4, 5])
parser.add_argument('--vocab_len', type=int, default=40000)
parser.add_argument('--output_channels', type=int, default=200)
parser.add_argument('--device', type=int, default=4)
parser.add_argument('--tokenizer', type=str, default='nltk', choices=('nltk', 'mose'))
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--train_or_test', type=str, default='train', choices=('train', 'test'))
args = parser.parse_args()

# 设置GPU
if torch.cuda.is_available():
    torch.cuda.set_device(args.device)

# 定义模型
class TextCNN(nn.Module):
    def __init__(self, pretrain_vectors, vocab_len, embedding_dim, output_channels, window_size):
        '''
        参数列表中的window_size是一个数组，里面每个值代表一个卷积核的window_size
        '''
        super(TextCNN, self).__init__()
        self.dynamic_embed = nn.Embedding(vocab_len, embedding_dim)
        self.dynamic_embed.weight.data.copy_(pretrain_vectors)
        self.dynamic_embed.weight.requires_grad = True
        self.static_embed = nn.Embedding(vocab_len, embedding_dim)
        self.static_embed.weight.data.copy_(pretrain_vectors)
        self.static_embed.weight.requires_grad = False
        self.filter_num = len(window_size)
        self.conv_layer = nn.ModuleList(nn.Conv1d(embedding_dim * 2, output_channels, ws) for ws in window_size)
        # self.batchnorm = nn.BatchNorm1d(1)
        self.dropout = nn.Dropout(0.5)
        self.linear_layer = nn.Linear(self.filter_num * output_channels, 2)

    def forward(self, x):
        dyn_embeds = self.dynamic_embed(x)  # 64 x 300 x 200
        sta_embeds = self.static_embed(x)
        embeds = torch.cat((dyn_embeds, sta_embeds), 2).permute(0, 2, 1) # 64 x 400 x 300
        feature_maps = []
        for conv in self.conv_layer:
            conv_res = F.relu(conv(embeds))                             # 64 x 100 x 298/297/296
            maxpool_res = F.max_pool1d(conv_res, conv_res.shape[2])  
            feature_maps.append(maxpool_res)
        feature = torch.cat(feature_maps, 1)
        feature = self.dropout(feature).flatten(1)
        output = self.linear_layer(feature)
        return output

data_path = 'processed_data_' + str(args.sentence_len) + '_' + args.tokenizer + '.pkl'
if args.train_or_test == 'train':
    if not os.path.exists(data_path):
        # 读取数据集中的句子和标签
        all_words = []
        train_data = []
        train_label = []
        test_data = []
        test_label = []
        print("data hasn't been saved, preprocessing data...")
        mt = MosesTokenizer()
        with open('train_data.txt', 'r', encoding='utf-8') as ftrain_data, \
            open('train_labels.txt', 'r', encoding='utf-8') as ftrain_label, \
            open('test_data.txt', 'r', encoding='utf-8') as ftest_data, \
            open('test_labels.txt', 'r', encoding='utf-8') as ftest_label:
            while True:
                sentence = ftrain_data.readline().strip().lower().replace('<br />', ' ').replace('/', ' / ')
                if not sentence:
                    break
                label = ftrain_label.readline().strip()
                if args.tokenizer == 'mose':
                    sentence = mt.tokenize(sentence, return_str=True).split()
                else:
                    sentence = nltk.word_tokenize(sentence)
                train_data.append(sentence)
                train_label.append(0 if label == 'neg' else 1)
                all_words.extend(sentence)

            while True:
                sentence = ftest_data.readline().strip().lower().replace('<br />', ' ').replace('/', ' / ')
                label = ftest_label.readline().strip()
                if not sentence:
                    break
                if args.tokenizer == 'mose':
                    sentence = mt.tokenize(sentence, return_str=True).split()
                else:
                    sentence = nltk.word_tokenize(sentence)
                test_data.append(sentence)
                test_label.append(0 if label == 'neg' else 1)

        # 构建词表
        word_counter = Counter(all_words)
        vocab_len = args.vocab_len - 2  
        most_common_words = word_counter.most_common(vocab_len)
        most_common_words = ['UNK', 'PAD'] + [word[0] for word in most_common_words]
        vocab_len = args.vocab_len
        vocab = defaultdict(lambda : 0)
        for idx, word in enumerate(most_common_words):
            vocab[word] = idx
        with open('vocab.pkl', 'wb') as f:
            pickle.dump(most_common_words, f)
        # 加载GloVe词向量
        wv_model = KeyedVectors.load('w2v_model.model')
        pretrained_vectors = np.random.rand(vocab_len, args.embedding_dim) - 0.5
        for i in range(2, vocab_len):
            if most_common_words[i][0] in wv_model:
                pretrained_vectors[i, :] = wv_model[most_common_words[i][0]]
        pretrained_vectors = torch.from_numpy(pretrained_vectors).float()

        # 将每个句子当中的词转换为词表当中的索引
        max_sentence_len = args.sentence_len
        train_sentence_idx = []
        test_sentence_idx = []

        for sentence in train_data:
            s = []
            for word in sentence:
                if len(s) >= args.sentence_len:
                    break
                s.append(vocab[word])
            s.extend([vocab['PAD'] for i in range(args.sentence_len - len(sentence))])
            train_sentence_idx.append(s)
        for sentence in test_data:
            s = []
            for word in sentence:
                if len(s) >= args.sentence_len:
                    break
                s.append(vocab[word])  
            s.extend([vocab['PAD'] for i in range(args.sentence_len - len(sentence))])
            test_sentence_idx.append(s)

        train_sentence_idx = torch.Tensor(train_sentence_idx).long()
        train_label = torch.Tensor(train_label).long()
        test_sentence_idx = torch.Tensor(test_sentence_idx).long()
        test_label = torch.Tensor(test_label).long()

        train_dataset = Data.TensorDataset(train_sentence_idx, train_label)
        val_dataset = Data.TensorDataset(test_sentence_idx, test_label)

        with open(data_path, 'wb') as f:
            data = {
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'wordvector': pretrained_vectors
            }
            pickle.dump(data, f)
        print("data save finished!")

    else:
        print('data is already preprocessed, loading...')
        with open(data_path, 'rb') as f:
            datas = pickle.load(f)
            train_dataset = datas['train_dataset']
            val_dataset = datas['val_dataset']
            pretrained_vectors = datas['wordvector']
        print('data is ready!')

    train_set = Data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True)
    val_set = Data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True)

else:
    # 处理测试集数据
    test_data_path = 'unlabel_data_' + str(args.sentence_len) + '.pkl'
    if not os.path.exists(test_data_path):
        test_data = []
        with open('test.txt', 'r', encoding='utf-8') as f:
            while True:
                sentence = f.readline()
                if not sentence:
                    break
                sentence = sentence.strip().lower().replace('<br />', ' ').replace('/', ' / ')
                sentence = nltk.word_tokenize(sentence)
                test_data.append(sentence)
        with open('vocab.pkl', 'rb') as f:
            most_common_words = pickle.load(f)
        vocab = defaultdict(lambda : 0)
        for idx, word in enumerate(most_common_words):
            vocab[word] = idx
        test_sentence_2idx = []
        for sentence in test_data:
            sentence_to_idx = []
            for word in sentence:
                if len(sentence_to_idx) < args.sentence_len:
                    sentence_to_idx.append(vocab[word])
                else:
                    break
            sentence_to_idx.extend([vocab['PAD'] for i in range(args.sentence_len - len(sentence_to_idx))])
            test_sentence_2idx.append(sentence_to_idx)
        test_data = torch.Tensor(test_sentence_2idx).long()
        with open(test_data_path, 'wb') as fw:
            pickle.dump(test_data, fw)
    else:
        with open(test_data_path, 'rb') as fr:
            test_data = pickle.load(fr)
    print('data is already preprocessed, loading...')
    with open(data_path, 'rb') as f:
        datas = pickle.load(f)
        # train_dataset = datas['train_dataset']
        # val_dataset = datas['val_dataset']
        pretrained_vectors = datas['wordvector']
    print('data is ready!')


model_para = '{}+{}+{}+{}+{}'.format(args.tokenizer, args.vocab_len, args.sentence_len, args.window_size, args.output_channels)
model_path = 'models/{}.bin'.format(model_para)
# 定义模型和优化器
model = TextCNN(pretrained_vectors, args.vocab_len, args.embedding_dim, args.output_channels, args.window_size).cuda()
if os.path.exists(model_path):
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)

if args.train_or_test == 'train':
    optim = opt.Adam(model.parameters(), lr=args.lr)
    loss_func = nn.CrossEntropyLoss()
    best_acc = 0.875
    # 模型训练和验证
    print('start training...')
    for ep in range(args.epoch):
        model.train()
        train_loss = []
        train_correct_cnt = 0
        for step, (train_x, train_y) in enumerate(train_set):
            model.zero_grad()
            train_x = train_x.cuda()
            train_y = train_y.cuda()
            output = model(train_x)
            loss = loss_func(output, train_y)
            train_loss.append(loss.item())
            train_pred = torch.argmax(output, dim=1)
            train_correct_cnt += train_pred.eq(train_y.view_as(train_pred)).sum().item()
            loss.backward()
            optim.step()
        train_loss = sum(train_loss) / step
        train_acc = 1.0 * train_correct_cnt / len(train_set.dataset)
        
        model.eval()
        correct_n = 0
        val_loss = []
        for val_x, val_y in val_set:
            val_x = val_x.cuda()
            val_y = val_y.cuda()
            val_output = model(val_x)
            val_loss.append(F.cross_entropy(val_output, val_y).item()) 
            val_pred = torch.argmax(val_output, dim=1)
            correct_n += val_pred.eq(val_y.view_as(val_pred)).sum()
        val_loss = sum(val_loss) / len(val_loss)
        val_acc = 1.0 * correct_n / len(val_set.dataset)
        print('epoch = %d, loss = %.4f, acc = %.4f, val_loss = %.4f, val_acc = %.4f'%(ep, train_loss, train_acc, val_loss, val_acc))
        if val_acc >= best_acc:
            best_acc = val_acc
            if not os.path.exists('models/'):
                os.mkdir('models/')
            torch.save(model.state_dict(), model_path)

else:
    # 验证模型是否可用
    with open("processed_data_600_nltk.pkl", 'rb') as f:
        val_dataset = pickle.load(f)['val_dataset']
    val_set = Data.DataLoader(val_dataset, batch_size=args.batchsize, shuffle=True)
    model.eval()
    cn = 0
    for val_x, val_y in val_set:
        val_x = val_x.cuda()
        val_y = val_y.cuda()
        output = model(val_x)
        pred = torch.argmax(output, 1)
        cn += pred.eq(val_y.view_as(pred)).sum().item()
    print("acc on dev set = {}".format(1.0 * cn / len(val_set.dataset)))
    print('no problem in model')

    #写入测试结果
    print('start testing...')
    with open('vocab.pkl', 'rb') as f:
        most_common_words = pickle.load(f)
    test_data = test_data.cuda()
    test_pred = []
    for i in range(0, test_data.shape[0] - 99, 100):
        test_output = model(test_data[i: i+100])
        test_pred.extend(torch.argmax(test_output, dim=1).tolist())
    with open('result.txt', 'w', encoding='utf-8') as result:
        for p in test_pred:
            result.write('pos\n' if p == 1 else 'neg\n')
    print('test result writting over')
    