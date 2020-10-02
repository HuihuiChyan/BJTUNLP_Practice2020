import re 
import os

import numpy as np
from itertools import chain

#返回数组，分别为文本+特征
def read_files(path,filetype):
	file_list = []
	pos_path = path + filetype + "/pos/"
	neg_path = path + filetype + "/neg/"
	for f in os.listdir(pos_path):
		file_list += [[pos_path+f,1]]
	for f in os.listdir(neg_path):
		file_list +=[[neg_path+f,0]]
	data = []
	for fi,label in file_list:
		with open(fi,encoding='utf8') as fi:
			data += [[" ".join(fi.readlines()),label]]
	return data

def get_stop_words_list(filepath):
	stop_words_list = []
	with open(filepath,encoding='utf8') as f:
		for line in f.readlines():
			stop_words_list.append(line.strip())
	return stop_words_list
def data_process(text):
	re_tag = re.compile(r'[^[a-z\s]')
	text.lower()
	text = re_tag.sub('',text)
	text = " ".join([word for word in text.split(' ')])
	return text
#进行分词+转换为小些
def get_token_text(text):
	#token_data = [data_process(st) for st in text.split()]
	token_data = [st.lower() for st in text.split()]
	token_data = list(filter(None,token_data))
	return token_data
#返回文本分词形式
def get_token_data(data):
	data_token = []
	for st,label in data:
		data_token.append(get_token_text(st))
	return data_token
def get_vocab(data):
    #将分词放入set，不重复。类似建立语料库
	vocab = set(chain(*data))
	vocab_size = len(vocab)
    #建立语料库和索引
	word_to_idx  = {word: i+1 for i, word in enumerate(vocab)}
	word_to_idx['<unk>'] = 0
	idx_to_word = {i+1: word for i, word in enumerate(vocab)}
	idx_to_word[0] = '<unk>'
	return vocab,vocab_size,word_to_idx,idx_to_word
def encode_st(token_data,vocab,word_to_idx):
	features = []
	for sample in token_data:
		feature = []
		for token in sample:
			if token in word_to_idx:
				feature.append(word_to_idx[token])
			else:
				feature.append(0)
		features.append(feature)
	return features
#填充和截断
def pad_st(features,maxlen,pad=0):
	padded_features = []
	for feature in features:
		if len(feature)>maxlen:
			padded_feature = feature[:maxlen]
		else:
			padded_feature = feature
			while(len(padded_feature)<maxlen):
				padded_feature.append(pad)
		padded_features.append(padded_feature)
	return padded_features
data_path = "/Users/ren/jupyter/NLP作业/data1/aclImdb/"
save_path = ""
maxlen = 300
train_data = read_files(data_path,"train")
test_data = read_files(data_path,"test")
print("read_file success!")
print(train_data[0][0],test_data[0][1])
train_token = get_token_data(train_data)
test_token = get_token_data(test_data)
print("get_token_data success!")
print(train_token[0])
vocab,vocab_size,word_to_idx,idx_to_word = get_vocab(train_token)
np.save("vocab.npy",vocab)

print("vocab_save success!")
train_features = pad_st(encode_st(train_token, vocab,word_to_idx),maxlen)
test_features = pad_st(encode_st(test_token, vocab,word_to_idx),maxlen)
train_label = [score for _, score in train_data]
test_label = [score for _, score in test_data]
print("get_feature_data success!")
#.npy文件是numpy专用的二进制文件，np.save和np.load
np.save("train_features.npy",train_features)
np.save("test_features.npy",test_features)
np.save("train_label.npy",train_label)
np.save("test_label.npy",test_label)