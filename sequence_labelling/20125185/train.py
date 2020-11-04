import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from model import BiLSTM_CRF
import data_process
import argparse
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

torch.manual_seed(1)
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=40000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--embedding_dim', type=int, default=300)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--test_per_step', type=int, default=100)
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--model_path', type=str, default='./model/best_f1.bin')

args = parser.parse_args()

label2idx = {'O':0, 'B-LOC':1, 'B-PER':2, 'B-ORG':3, 'I-PER':4, 'I-ORG':5, 'B-MISC':6, 'I-LOC':7, 'I-MISC':8, 'START':9, 'STOP':10}
idx2label = ['O', 'B-LOC', 'B-PER', 'B-ORG', 'I-PER', 'I-ORG', 'B-MISC', 'I-LOC', 'I-MISC', 'START', 'STOP']

# 加载数据
train_iter, valid_iter, test_iter, text_vocab = data_process.load_data(args.batch_size,device)

args.vocab_size = len(text_vocab)
print('vocab_size:',len(text_vocab))

# 加载预训练词向量
weight = text_vocab.vectors
# with open('./data/wvmodel.pkl', 'rb') as inp:
#     wvmodel = pickle.load(inp)
# print('wvmodel loaded!')
#
# weight = torch.zeros(args.vocab_size, args.embedding_size)
# for i in range(len(wvmodel.index2word)):
#     try:
#         index = word_to_idx[wvmodel.index2word[i]]
#     except:
#         continue
#     weight[index,:] = torch.from_numpy(wvmodel.get_vector(
#         idx_to_word[word_to_idx[wvmodel.index2word[i]]]))


model = BiLSTM_CRF(args, label2idx, weight,device).to(device)
optimizer = optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

best_f1 = 0.0
print('training on ',device)
for epoch in range(1):
    model.train()
    train_loss_sum = 0.0
    steps = 0
    for batch in train_iter:
        X, y = batch.TEXT, batch.LABEL
        X, y = X.to(device).long(), y.to(device).long()

        loss = model.neg_log_likelihood(X, y)

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()))
        optimizer.step()

        train_loss_sum += loss
        steps+=1

        # 测试
        if steps % test_per_step ==0:
            model.eval()
            valid_loss_sum = 0.0
            m = 0
            y_pre,y_true=[],[]
            for batch in valid_iter:
                X, y = batch.TEXT, batch.LABEL
                X, y = X.to(device).long(), y.to(device).long()

                _, y_hat = model(X)
                loss_t = model.neg_log_likelihood(X, y)
                valid_loss_sum += loss_t
                m = y.shape[0]

                y_pre.append(y_hat)
                y_true.append(y.squeeze(1)).tolist()

            y_pre = [[idx2label[idx] for idx in y_pre_idx] for y_pre_idx in y_pre]
            y_true = [[idx2label[idx] for idx in y_true_idx] for y_true_idx in y_true]
            #print(y_pre[10],y_true[10])

            # 评价指标
            P = precision_score(y_true, y_pre)
            R = recall_score(y_true, y_pre)
            F1 = f1_score(y_true, y_pre)

            if F1 > best_f1:
                best_f1 = F1
                torch.save(model.state_dict(), args.model_path)

            print('train_step %d,train_loss %.4f, P %.3f, R %.3f, F1 %.4f'%(steps,train_loss_sum%n,P,R,F1))



# print('test')
# model = BiLSTM_CRF(args, label2idx, weight,device).to(device)
# model.load_state_dict(torch.load('./model/best_f1.bin'))
# model.eval()
# m = 0
# y_pre,y_true=[],[]
# for batch in valid_iter:
#     X, y = batch.TEXT, batch.LABEL
#     X, y = X.to(device).long(), y.to(device).long()
#     _, y_hat = model(X)
#     y_pre.append(y_hat)
#     y_true.append(y.squeeze(1)).tolist()
# y_pre = [[idx2label[idx] for idx in y_pre_idx] for y_pre_idx in y_pre]
# y_true = [[idx2label[idx] for idx in y_true_idx] for y_true_idx in y_true]
# with open("./data/result.txt",'w',encoding='utf-8') as f:
#     for line in pred:
#         f.write(' '.join(line)+'\n')










