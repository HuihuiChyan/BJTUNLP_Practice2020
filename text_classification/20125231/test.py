import glob
import os
import torch
import torch.nn.functional as F
from torch import nn
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
mt = MosesTokenizer(lang='en')
parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=121339)
parser.add_argument('--embedding_size', type=int, default=512)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--hidden_size', type=int, default=1024)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--epoch_num', type=int, default=100)
parser.add_argument('--steps_per_log', type=int, default=10)
parser.add_argument('--steps_per_eval', type=int, default=20)
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()

wordls = {}
with open('words.txt','r',encoding='utf-8') as wordfile:
	i = 0
	for w in wordfile.readlines():
		w = w.strip()
		wordls[w] = i
		i+=1

all_test = []

with open('test(1).txt','r',encoding='utf-8') as fin:
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
		all_test.append(line_index)
all_test = torch.tensor(all_test)
test_dataset = torch.utils.data.TensorDataset(all_test)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
										  batch_size=args.batch_size,
										  shuffle=False)
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

fc_model = FC(args)
fc_model.load_state_dict(torch.load('params.bin'))
print('model initialized from params.bin')
fc_model.eval()

all_test_logits = []
for batch_test_text in test_loader:
	batch_test_output = fc_model(batch_test_text[0])
	batch_test_logits = torch.argmax(batch_test_output, dim=-1)
	all_test_logits.extend(batch_test_logits)
with open('result.txt', 'w', encoding='utf-8') as fresult:
	for logit in all_test_logits:
		if logit.tolist() == 0:
			fresult.write('neg' + '\n')
		elif logit.tolist() == 1:
			fresult.write('pos' + '\n')
		else:
			raise Exception('why extra label?')

