import torch
import os
import pdb
import pickle
import nltk
import os
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# 获取文件名列表
def get_file_list(path):
    file_list = []
    for file_name in os.listdir(path):
        file_list.append(path + file_name)
    return file_list

# 从文件中读取数据
def read_datas(pos,neg):
    texts, labels = [],[]
    # pos类别---1
    for file_name in pos: 
        with open(file_name,'r') as fr:
            texts.append(fr.readline())
            labels.append(1)
    # neg类别---0
    for file_name in neg:
        with open(file_name,'r') as fr: 
            texts.append(fr.readline())
            labels.append(0)
    return texts,labels


# 将分词后的结果存储到pkl中,方便加载使用
def write_token():
	train_pos = get_file_list('data/aclImdb/train/pos/')
	train_neg = get_file_list('data/aclImdb/train/neg/')
	test_pos = get_file_list('data/aclImdb/test/pos/')
	test_neg = get_file_list('data/aclImdb/test/neg/')
	train_texts,train_labels = read_datas(train_pos,train_neg)
	test_texts,test_labels = read_datas(test_pos,test_neg)

	# 分词
	train_texts = [nltk.word_tokenize(text) for text in train_texts]
	train_texts = [[word.lower() for word in text] for text in train_texts]
	test_texts = [nltk.word_tokenize(text) for text in test_texts]
	test_texts = [[word.lower() for word in text] for text in test_texts]

	with open('./data/data.pkl', 'wb') as outp:
	    pickle.dump(train_texts, outp)
	    pickle.dump(train_labels, outp)
	    pickle.dump(test_texts, outp)
	    pickle.dump(test_labels, outp)

	print('tokenized data has been saved!')

# write_token()


# 加载预训练词向量glove，并转换为word2vec模型，保存在wvmodel中，方便加载使用
def save_wvmodel():
	
	if os.path.exists('./data/word2vec.txt'):
	    wvmodel = KeyedVectors.load_word2vec_format('./data/word2vec.txt',binary=False,encoding='utf-8')
	else:
	    path=os.getcwd()
	    glove_file=datapath(os.path.join(path,'data/glove.6B.300d.txt'))
	    tmp_file=get_tmpfile(os.path.join(path,"data/word2vec.txt"))
	    glove2word2vec(glove_file, tmp_file)
	    wvmodel = KeyedVectors.load_word2vec_format(tmp_file,binary=False,encoding='utf-8')

	with open('./data/wvmodel.pkl', 'wb') as outp:
	    pickle.dump(wvmodel, outp)

#save_wvmodel()






