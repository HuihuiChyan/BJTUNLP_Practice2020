import os
import torch
import torch.nn as nn
# import gensim
# from gensim.models import Word2Vec
from collections import defaultdict,Counter
import pdb 
import argparse
import torch.nn.functional as F
from sacremoses import MosesTokenizer

# 参数列表
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=80000)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--output_dim',type=int,default=2)
parser.add_argument('--kernel_size',type=list,default=[2,3,4,5])
parser.add_argument('--kernel_num',type=int,default=256)#每种卷积和的大小
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--param_path',type=str, default='./dataset/param.bin')
parser.add_argument('--test_path',type=str, default='./dataset/test2.txt')
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()

# 模型
class TextCNN(nn.Module):
    def __init__(self,args):
        super(TextCNN, self).__init__()

        self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)

        self.convs = nn.ModuleList(
            [nn.Conv2d(1, args.kernel_num, (k, args.embedding_size)) for k in args.kernel_size]
        )
        
        self.fc = nn.Linear(len(args.kernel_size) * args.kernel_num, args.output_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, text):
        x = self.embedding(text)
        
        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] # len(Ks)*(N,Knum,W)
        
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        
        #(batch,n_filters)

        cat = self.dropout(torch.cat(x, dim=1))

        return self.fc(cat)

def loadData(tag):
    """
    tag: train/test
    读取数据
    """
    url = './dataset/'+ tag +'.txt'
    with open(url, encoding='utf-8') as file:
        line = file.readlines()
        all_data = [item.strip('\n').split() for item in line]
    label = [data[0] for data in all_data]
    feature = [data[1:] for data in all_data]
    return feature, label

def train(args, train_loader, model,optim,criterion):
    """
    训练函数
    """
    loss_log = []
    global_step = 0
    best_eval_acc = 0.0
    for epoch in range(args.epoch_num):
        total_acc,total_loss,correct,sample_num= 0, 0, 0,0
        for feature, batch_labels in train_loader:
            feature = feature.cuda()
            batch_labels = batch_labels.cuda()

            model.train()
            optim.zero_grad()
            
            y_hat = model(feature)
            
            
            loss = criterion(y_hat,batch_labels)

            loss.backward()
            optim.step()
            global_step += 1
            correct = (y_hat.argmax(dim=1) == batch_labels).float().sum().item()
            total_acc += correct
            sample_num += len(y_hat.argmax(dim=1))
            total_loss += loss.item()
            loss_log.append(loss.item())
        if global_step % args.steps_per_log == 0:
            print('Train {:d}| Loss:{:.5f} Acc: {:.4f}'.format(epoch+1 ,total_loss / len(train_loader), total_acc/ sample_num))
        if global_step % args.steps_per_eval == 0:
            test_acc, test_loss = evaluate_accuracy(eval_loader,model, criterion)
            print('at train step %d, eval accuracy is %.4f, eval loss is %.4f' % (global_step, test_acc, test_loss))

            if test_acc > best_eval_acc:
                best_eval_acc = test_acc
                torch.save(model.state_dict(), args.param_path)

def evaluate_accuracy(data_iter, net,criterion):
    """
    评估准确率
    """
    net.eval()
    acc_sum, n ,loss_sum= 0.0, 0, 0.0
    epoch = 0
    for X, y in data_iter:
        epoch += 1
        X = X.cuda()
        y = y.cuda()
        y_hat = net(X)
        
        acc_sum += (y_hat.argmax(dim=1) == y).float().sum().item()
        loss = criterion(y_hat, y).sum()
        loss_sum = loss.item()
        n += y.shape[0]
        # pdb.set_trace()

    return acc_sum / n, loss_sum / epoch 

if args.train_or_test == 'train':
    print('train begin')
    #load训练数据和测试数据
    train_feature, train_label = loadData('train')
    test_feature, test_label = loadData('test')

    #生成词表 begin
    if os.path.exists('./dataset/vocab.txt'):
        with open('./dataset/vocab.txt',"r",encoding='utf-8') as fvocab:
            vocab_words = [line for line in fvocab]
    else:
        train_word = []
        for line in train_feature:
            train_word.extend(line)
            
        counter = Counter(train_word)
        
        common_words = counter.most_common()

        vocab_words = [word[0] for word in common_words[:args.vocab_size-2]]
        
        vocab_words = ['[UNK]','[PAD]'] + vocab_words

        with open("./dataset/vocab.txt","w",encoding='utf-8') as fvocab:
            for word in vocab_words:
                fvocab.write(word+'\n')

    # 缺省值 [UNK] 填充值 [PAD]
    word2idx = defaultdict(lambda :0)	
    idx2word = defaultdict(lambda:'[UNK]')

    for idx,word in enumerate(vocab_words):
        word2idx[word.strip('\n')] = idx
        idx2word[idx] = word.strip('\n')
    # 词表生成 end

    train_textlines = [line[:args.max_len] for line in train_feature]
    test_textlines = [line[:args.max_len] for line in test_feature]

    train_textlines = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in train_textlines]
    test_textlines = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in test_textlines]
    # 转化为词表里面的index
    train_textlines = [[word2idx[word] for word in line] for line in train_textlines]
    test_textlines = [[word2idx[word] for word in line] for line in test_textlines]

    train_labellines = [1 if label=='neg' else 0 for label in train_label]
    test_labellines = [1 if label=='neg' else 0 for label in test_label]

    # pdb.set_trace()

    train_textlines = torch.tensor(train_textlines)
    train_labellines = torch.tensor(train_labellines)

    test_textlines = torch.tensor(test_textlines)
    test_labellines = torch.tensor(test_labellines)

    train_dataset = torch.utils.data.TensorDataset(train_textlines, train_labellines)
    test_dataset = torch.utils.data.TensorDataset(test_textlines, test_labellines)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True)

    eval_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)


    model = TextCNN(args).cuda()

    if os.path.exists(args.param_path):
        print('loading params')
        # pdb.set_trace()
        model.load_state_dict(torch.load(args.param_path))

    optim = torch.optim.Adam(model.parameters(), args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train(args, train_loader,model, optim, criterion)
else:
    print('test begin')
    with open(args.test_path, 'r', encoding='utf-8') as ftest_text:
        test_textlines = [line.strip() for line in ftest_text.readlines()]
        mt = MosesTokenizer(lang='en')

        test_textlines = [mt.tokenize(line, return_str=True) for line in test_textlines]
        test_textlines = [line.lower() for line in test_textlines]

        if os.path.exists('./dataset/vocab.txt'):
            with open('./dataset/vocab.txt', 'r', encoding='utf-8') as fvocab:
                vocab_words = [line.strip() for line in fvocab.readlines()]
        else:
            raise Exception('no vocabulary')
        
        word2idx = defaultdict(lambda :0)

        for idx,word in enumerate(vocab_words):
            word2idx[word] = idx

		# 最大句长卡到300
        test_textlines = [line.split()[:args.max_len] for line in test_textlines]
        test_textlines = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in test_textlines]
        test_textlines = [[word2idx[word] for word in line] for line in test_textlines]
        test_textlines = torch.tensor(test_textlines)
        test_dataset = torch.utils.data.TensorDataset(test_textlines)

        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=False)

        model = TextCNN(args).cuda()

        if os.path.exists(args.param_path):
            model.load_state_dict(torch.load(args.param_path))
            print('model initialized from params.bin')
        else:
            raise Exception('no params.bin?')

        model.eval()

        all_test_logits = []
        for batch_test_text in test_loader:
            batch_test_text = batch_test_text[0].cuda()
            
            batch_test_output = model(batch_test_text)
            batch_test_logits = torch.argmax(batch_test_output, dim=-1)
            all_test_logits.extend(batch_test_logits)

        with open('./dataset/result.txt', 'w', encoding='utf-8') as fresult:
            for logit in all_test_logits:
                if logit.tolist() == 1:
                    fresult.write('neg' + '\n')
                elif logit.tolist() == 0:
                    fresult.write('pos' + '\n')

            print('test end')