import torch
import os
import pdb
import pickle
import nltk
from torch.utils import data
from collections import Counter,defaultdict



# 建立词表
def set_vocab(train_texts):	
	if os.path.exists('./data/vocab.txt'):  # 如果词表存在
		with open('./data/vocab.txt','r',encoding='utf-8') as fr:
			vocab_words = [line.strip() for line in fr.readlines()]
	else:
		train_words = []
		for text in train_texts:
			train_words.extend(text)

		# 统计词频
		common_words = Counter(train_words).most_common()
		vocab_words = ['[UNK]','[PAD]']+[word[0] for word in common_words]
		# 写入词表
		fw = open('./data/vocab.txt','w',encoding='utf-8')
		for word in vocab_words:
			fw.write(word+'\n')
	print('vocab loaded!')
	return vocab_words


# 加载数据,生成迭代器
def load_data(args):
	print('loading datas......')
	with open('./data/data.pkl', 'rb') as inp:
	    train_texts = pickle.load(inp)
	    train_labels = pickle.load(inp)
	    test_texts = pickle.load(inp)
	    test_labels = pickle.load(inp)

	if args.pretrained:  # 加载预训练词向量，使用预训练词向量的词表
		with open('./data/wvmodel.pkl', 'rb') as inp:
		    wvmodel = pickle.load(inp)
		vocab_words = list(wvmodel.vocab.keys())
	else:
		vocab_words = set_vocab(train_texts)

	# word转换为idx
	word2idx = defaultdict(lambda :0)  #默认为0---UNK
	for idx, word in enumerate(vocab_words):
		word2idx[word] = idx 


	# 取最大句长，不足的padding
	train_texts = [line[:args.max_len] for line in train_texts]
	test_texts = [line[:args.max_len] for line in test_texts]
	train_texts = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in train_texts]
	test_texts = [line + ['[PAD]' for i in range(args.max_len-len(line))] for line in test_texts]

	# 生成数据集,每句话对应一串数字
	train_datas = [[word2idx[word] for word in text] for text in train_texts]
	test_datas = [[word2idx[word] for word in text] for text in test_texts]

	train_datas = torch.tensor(train_datas)
	test_datas = torch.tensor(test_datas)
	train_labels = torch.tensor(train_labels)
	test_labels = torch.tensor(test_labels)


	train_datasets = data.TensorDataset(train_datas,train_labels)
	test_datasets = data.TensorDataset(test_datas,test_labels)

	# 生成迭代器
	train_iter = data.DataLoader(train_datasets,args.batch_size,shuffle=True,num_workers=2)
	test_iter = data.DataLoader(test_datasets,args.batch_size,shuffle=True,num_workers=2)


	print('datas loaded!')
	return train_iter,test_iter,vocab_words


  