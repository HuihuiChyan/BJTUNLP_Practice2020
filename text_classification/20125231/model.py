import glob
import os
import torch
import argparse
import torch.nn.functional as F
from torch import nn
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=121339)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--steps_per_eval', type=int, default=20)
args = parser.parse_args()

from sacremoses import MosesTokenizer, MosesDetokenizer
mt = MosesTokenizer(lang='en')

wordls = {}
with open('words.txt','r',encoding='utf-8') as wordfile:
	i = 0
	for w in wordfile.readlines():
		w = w.strip()
		wordls[w] = i
		i+=1

all_train = []
all_train_label = []
all_test = []
all_test_label = []
with open('train_negs.txt','r',encoding='utf-8') as fin:
	for line in fin.readlines():
		line = line.strip()
		line = mt.tokenize(line, return_str=True)
		line_index = []
		for words in line.split()[:300]:
			if words not in wordls:
				words = 'unk'
			line_index.append(wordls[words])
		for i in range(300 - len(line_index)):
			line_index.append(0)
		all_train_label.append(0) #0是负面
		all_train.append(line_index)
with open('train_poss.txt','r',encoding='utf-8') as fin:
	for line in fin.readlines():
		line = line.strip()
		line = mt.tokenize(line, return_str=True)
		line_index = []
		for words in line.split()[:300]:
			if words not in wordls:
				words = 'unk'
			line_index.append(wordls[words])
		for i in range(300-len(line_index)):
			line_index.append(0)
		all_train_label.append(1)  # 1是正面
		all_train.append(line_index)

with open('test_negs.txt','r',encoding='utf-8') as fin:
	for line in fin.readlines():
		line = line.strip()
		line = mt.tokenize(line, return_str=True)
		line_index = []
		for words in line.split()[:300]:
			if words not in wordls:
				words = 'unk'
			line_index.append(wordls[words])
		for i in range(300 - len(line_index)):
			line_index.append(0)
		all_test_label.append(0) #0是负面
		all_test.append(line_index)
with open('test_poss.txt','r',encoding='utf-8') as fin:
	for line in fin.readlines():
		line = line.strip()
		line = mt.tokenize(line, return_str=True)
		line_index = []
		for words in line.split()[:300]:
			if words not in wordls:
				words = 'unk'
			line_index.append(wordls[words])
		for i in range(300-len(line_index)):
			line_index.append(0)
		all_test_label.append(1)  # 1是正面
		all_test.append(line_index)

all_train = torch.tensor(all_train)
all_test = torch.tensor(all_test)
all_train_label = torch.tensor(all_train_label)
all_test_label = torch.tensor(all_test_label)

train_dataset = torch.utils.data.TensorDataset(all_train, all_train_label)
test_dataset = torch.utils.data.TensorDataset(all_test, all_test_label)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
										   batch_size=args.batch_size,
										   shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=args.batch_size,
										  shuffle=True)

class FC(nn.Module):
	def __init__(self, args):
		super(FC, self).__init__()
		num_kernel = 256
		kernel_sizes = [3, 4, 5]
		dropout = 0.2

		# self.a_p1 = nn.MaxPool1d(298)
		# self.a_p2 = nn.MaxPool1d(297)
		# self.a_p3 = nn.MaxPool1d(296)
		self.embedding = nn.Embedding(args.vocab_size, args.embedding_size)
		self.conv1 = nn.Conv2d(1, num_kernel, (kernel_sizes[0], args.embedding_size))
		self.conv2 = nn.Conv2d(1, num_kernel, (kernel_sizes[1], args.embedding_size))
		self.conv3 = nn.Conv2d(1, num_kernel, (kernel_sizes[2], args.embedding_size))
		self.ouptput_layer = nn.Linear(num_kernel*len(kernel_sizes), 2)


	def forward(self, batch_text):
		embeds = self.embedding(batch_text)#[batchsize,300,embeddingsize=512]
		flat_embeds = embeds.unsqueeze(1)#[batchsize,1,300,embeddingsize=512]
		num1 = self.conv1(flat_embeds)
		num2 = self.conv2(flat_embeds)
		num3 = self.conv3(flat_embeds)#[batchsize,num_kernel=256,296,1]
		num1 = num1.squeeze()
		num2 = num2.squeeze()
		num3 = num3.squeeze()#[batchsize,num_kernel=256,296]
		a = torch.max(num1, dim=2)[0]
		b = torch.max(num2, dim=2)[0]
		c = torch.max(num3, dim=2)[0]
		lala = torch.cat((a,b,c),1)
		output = self.ouptput_layer(lala)

		return output


fc_model = FC(args).to(device)

optim = torch.optim.Adam(fc_model.parameters(), args.learning_rate)
global_step = 0
best_eval_acc = 0.0
for epoch in range(args.epoch_num):
	for batch_text, batch_labels in train_loader:
		fc_model.train()#开启可训练状态
		optim.zero_grad()
		batch_text = batch_text.to(device)
		batch_labels = batch_labels.to(device)
		batch_output = fc_model(batch_text)
		loss = torch.nn.functional.cross_entropy(batch_output, batch_labels)
		loss.backward()
		optim.step()
		global_step += 1
		if global_step % args.steps_per_log == 0:
			print('train step %d, loss is %.4f' % (global_step, loss))
			fc_model.eval()#关闭可训练状态，开始验证
			all_eval_logits = []
			all_eval_labels = []
			for eval_batch_text, eval_batch_labels in eval_loader:
				eval_batch_text = eval_batch_text.to(device)
				eval_batch_labels = eval_batch_labels.to(device)
				eval_batch_output = fc_model(eval_batch_text)
				eval_batch_logits = torch.argmax(eval_batch_output, dim=-1)
				all_eval_logits.extend(eval_batch_logits.tolist())
				all_eval_labels.extend(eval_batch_labels.tolist())

			eval_acc = sum([int(line[0] == line[1]) for line in zip(all_eval_logits, all_eval_labels)]) / len(
				all_eval_labels)
			print('at train step %d, eval accuracy is %.4f' % (global_step, eval_acc))
			if eval_acc > best_eval_acc:
				best_eval_acc = eval_acc
				torch.save(fc_model.state_dict(), 'params.bin')

