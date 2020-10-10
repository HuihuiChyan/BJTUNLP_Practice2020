import numpy as np
import os
import torch
import torch.nn as nn
import _pickle as cPickle
import sys
import time
f = open('mapping.pkl', 'rb')
data = cPickle.load(f)
train_dataset = data['train_dataset']
test_dataset = data['test_dataset']
wordvector = data['wordvector']

max_len = 300
batch_size = 64
lr = 0.0003
num_epoch = 10

class TextCNN(nn.Module):
    def __init__(self, embed_size, out_channel, num_classes, k_sizes=[2, 3, 5]):
        super(TextCNN, self).__init__()

        self.embed_size = embed_size
        self.num_classes = num_classes

        self.embedding = nn.Embedding.from_pretrained(wordvector, padding_idx=0)
        self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_size, out_channels=out_channel, kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=max_len - h + 1)
            )
            for h in k_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_channel * len(k_sizes), num_classes)

    def forward(self, x):
        embed_out = self.embedding(x)
        embed_out = embed_out.permute(0, 2, 1)

        conv_out = [conv(embed_out) for conv in self.convs]
        out = torch.cat(conv_out, dim=1)

        out = out.view(-1, out.size(1))
        fc_out = self.fc(self.dropout(out))

        return fc_out


def train(model, train_iter, optimizer, loss_func):
    model.train()
    epoch_loss = 0
    correct, sample_num = 0, 0
    for x_iter, y_iter in train_iter:
        optimizer.zero_grad()
        output = model(x_iter)
        loss = loss_func(output, y_iter)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        prediction = torch.argmax(output, 1)
        correct += (prediction == y_iter).sum().item()
        # 累加样本和
        sample_num += len(prediction)
        
    return epoch_loss / len(train_iter), correct / sample_num


def evaluate(model, test_iter, loss_func):
    model.eval()
    epoch_loss = 0
    correct, sample_num = 0, 0
    with torch.no_grad():
        for x_iter, y_iter in test_iter:
            output = model(x_iter)
            loss = loss_func(output, y_iter)

            epoch_loss += loss.item()

            prediction = torch.argmax(output, 1)
            correct += (prediction == y_iter).sum().item()
            sample_num += len(prediction)
    return epoch_loss / len(test_iter), correct / sample_num


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = TextCNN(embed_size=300, out_channel=300, num_classes=2)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    train_losses = []
    test_losses = []
    train_acces = []
    test_acces = []
    best_test_loss = float('inf')
    for epoch in range(num_epoch):
        start = time.time()

        train_loss, train_acc = train(model, train_iter, optimizer, loss_func)
        test_loss, test_acc = evaluate(model, test_iter, loss_func)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_acces.append(train_acc)
        test_acces.append(test_acc)
        end = time.time()

        epoch_mins, epoch_secs = epoch_time(start, end)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'my_model.pth')  # 只保存模型参数

        print(f'Epoch: {epoch + 1:2} | Time: {epoch_mins}分 {epoch_secs}秒')
        print(f'\tTrain Loss: {train_loss:.3f} | Train ACC: {train_acc:.3f}')
        print(f'\t Val. Loss: {test_loss:.3f} |  Val. ACC: {test_acc:.3f}')