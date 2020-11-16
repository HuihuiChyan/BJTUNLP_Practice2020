import torch
from pytorch_transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, TensorDataset
import argparse
import datetime
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', type=str, default='cuda:1')
parser.add_argument('--num_epoch', type=int, default=50)
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--ckp', type=str, default='ckp/model_1.pt')
parser.add_argument('--max_acc', type=float, default=0.8)
args = parser.parse_args()

if torch.cuda.is_available():
    print("using cuda......")
    device = torch.device(args.cuda)

with open('train_pos.txt', 'r', encoding='utf-8') as ftrain_feature:
    train_pos = ['[CLS] '+line.strip()+' [SEP]' for line in ftrain_feature.readlines()]
with open('train_neg.txt', 'r', encoding='utf-8') as ftrain_feature:
    train_neg = ['[CLS] '+line.strip()+' [SEP]' for line in ftrain_feature.readlines()]
train_feature_line = train_pos + train_neg
train_label_line = [1] * int(len(train_feature_line)/2)
temp = [0] * int(len(train_feature_line)/2)
train_label_line.extend(temp)

#读入验证集
with open('valid_pos.txt', 'r', encoding='utf-8') as ftest_feature:
    test_pos = ['[CLS] '+line.strip()+' [SEP]' for line in ftest_feature.readlines()]
with open('valid_neg.txt', 'r', encoding='utf-8') as ftest_feature:
    test_neg = ['[CLS] '+line.strip()+' [SEP]' for line in ftest_feature.readlines()]
test_feature_line = test_pos + test_neg
test_label_line = [1] * int(len(test_feature_line)/2)
temp = [0] * int(len(test_feature_line)/2)
test_label_line.extend(temp)

tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/vocab.txt')
train_feature = [tokenizer.tokenize(line) for line in train_feature_line]
test_feature = [tokenizer.tokenize(line) for line in test_feature_line]
train_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in train_feature]
test_feature_id = [tokenizer.convert_tokens_to_ids(line) for line in test_feature]
print('loading tokenizer...')

for j in range(len(train_feature_id)):
    # 将样本数据填充至长度为 512
    i = train_feature_id[j]
    if len(i) < 512:
        train_feature_id[j].extend([0]*(512 - len(i)))
    else:
        train_feature_id[j] = train_feature_id[j][0:511] + [train_feature_id[j][-1]]
for j in range(len(test_feature_id)):
    # 将样本数据填充至长度为 512
    i = test_feature_id[j]
    if len(i) < 512:
        test_feature_id[j].extend([0]*(512 - len(i)))
    else:
        test_feature_id[j] = test_feature_id[j][0:511] + [test_feature_id[j][-1]]

train_set = TensorDataset(torch.LongTensor(train_feature_id), torch.LongTensor(train_label_line))
train_loader = DataLoader(dataset=train_set, batch_size=4, shuffle=True)

test_set = TensorDataset(torch.LongTensor(test_feature_id), torch.LongTensor(test_label_line))
test_loader = DataLoader(dataset=test_set, batch_size=4, shuffle=False)

class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert, self).__init__()

        #modelConfig = BertConfig.from_pretrained('./bert-base-uncased/config.json')
        self.model = BertModel.from_pretrained('./bert-base-uncased/').to(device) #, config=modelConfig
        embedding_dim = self.model.config.hidden_size
        self.dropout = torch.nn.Dropout(0.1)
        self.fc = torch.nn.Linear(embedding_dim, 2)

    def forward(self, tokens, attention_mask):
        output = self.model(tokens, attention_mask=attention_mask)
        output = output[0][:, 0, :]   #output[0](batch size, sequence length, model hidden dimension)
        output = self.dropout(output)
        output = self.fc(output)
        return output

def test_evaluate(test_loader, model, max_acc):
    
    test_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    loss_func = torch.nn.CrossEntropyLoss()
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):

            out = model(data.to(device), attention_mask=(data>0).to(device))
            loss = loss_func(out, label.to(device))
            test_l += loss.item()
            n += 1

            prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
            label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()
            out_epoch.extend(prediction)
            label_epoch.extend(label)

            test_l += loss.item()
            n += 1

        acc = accuracy_score(label_epoch, out_epoch)
        if acc > max_acc : 
            max_acc = acc
            torch.save(model.state_dict(), args.ckp)
            print("save model......")

        return test_l/n, acc, max_acc

loss_func = torch.nn.CrossEntropyLoss()
model = Bert().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
max_acc = args.max_acc
print('start trainning....')
for epoch in range(args.num_epoch):
    
    train_l, n = 0.0, 0
    out_epoch, label_epoch = [], []
    start = datetime.datetime.now()

    for batch_idx, (data, label) in enumerate(train_loader):
        model.train()
        out = model(data.to(device), attention_mask=(data>0).to(device))
        
        loss = loss_func(out, label.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_l += loss.item()
        n += 1
        
        prediction = out.argmax(dim=1).data.cpu().numpy().tolist()
        label = label.view(1,-1).squeeze().data.cpu().numpy().tolist()
        out_epoch.extend(prediction)
        label_epoch.extend(label)
        
    train_acc = accuracy_score(label_epoch, out_epoch)
    train_loss = train_l/n
    test_loss, test_acc, max_acc = test_evaluate(test_loader, model, max_acc)
    end = datetime.datetime.now()

    print('epoch %d,train_loss %f, test_loss %f,  train_acc %f, test_acc %f, max_acc %f, time %s' % 
    (epoch+1, train_loss, test_loss, train_acc, test_acc, max_acc, end-start))





