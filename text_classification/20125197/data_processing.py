from collections import defaultdict, Counter
import os
import torch
import argparse
from sacremoses import MosesTokenizer
import _pickle as cPickle
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--vocab_size', type=int, default=80000)
parser.add_argument('--embedding_size', type=int, default=300)
parser.add_argument('--max_len', type=int, default=300)
parser.add_argument('--train_or_test', type=str, choices=('train', 'test'), default='train')
args = parser.parse_args()


if args.train_or_test == 'train':
    with open('train.txt', 'r', encoding='utf-8') as ftrain_text, \
            open('train.label', 'r', encoding='utf-8') as ftrain_label, \
            open('test.txt', 'r', encoding='utf-8') as ftest_text, \
            open('test.label', 'r', encoding='utf-8') as ftest_label:

        train_textlines = [line.strip() for line in ftrain_text.readlines()]
        train_labellines = [line.strip() for line in ftrain_label.readlines()]
        test_textlines = [line.strip() for line in ftest_text.readlines()]
        test_labellines = [line.strip() for line in ftest_label.readlines()]

        mt = MosesTokenizer(lang='en')

        train_textlines = [mt.tokenize(line, return_str=True) for line in train_textlines]
        test_textlines = [mt.tokenize(line, return_str=True) for line in test_textlines]

        train_textlines = [line.lower() for line in train_textlines]
        test_textlines = [line.lower() for line in test_textlines]

        if os.path.exists('vocab.txt'):
            with open('vocab.txt', 'r', encoding='utf-8') as fvocab:
                vocab_words = [line.strip() for line in fvocab.readlines()]

        else:
            train_words = []
            for line in train_textlines:
                train_words.extend(line.split())

            counter = Counter(train_words)
            common_words = counter.most_common()

            vocab_words = [word[0] for word in common_words[:args.vocab_size - 2]]  # [UNK], [PAD]

            vocab_words = ['[PAD]', '[UNK]'] + vocab_words

            with open('vocab.txt', 'w', encoding='utf-8') as fvocab:
                for word in vocab_words:
                    fvocab.write(word + '\n')

        word2idx = defaultdict(lambda: 1)
        idx2word = defaultdict(lambda: '[UNK]')
        for idx, word in enumerate(vocab_words):
            word2idx[word] = idx
            idx2word[idx] = word
        print("建立词表完成。")
        # 最大句长卡到300
        train_textlines = [line.split()[:300] for line in train_textlines]
        test_textlines = [line.split()[:300] for line in test_textlines]

        train_textlines = [line + ['[PAD]' for i in range(300 - len(line))] for line in train_textlines]
        test_textlines = [line + ['[PAD]' for i in range(300 - len(line))] for line in test_textlines]

        train_textlines = [[word2idx[word] for word in line] for line in train_textlines]
        test_textlines = [[word2idx[word] for word in line] for line in test_textlines]

        train_labellines = [1 if label == 'pos' else 0 for label in train_labellines]
        test_labellines = [1 if label == 'pos' else 0 for label in test_labellines]

        train_textlines = torch.tensor(train_textlines)
        train_labellines = torch.tensor(train_labellines)

        test_textlines = torch.tensor(test_textlines)
        test_labellines = torch.tensor(test_labellines)

        train_dataset = torch.utils.data.TensorDataset(train_textlines, train_labellines)
        test_dataset = torch.utils.data.TensorDataset(test_textlines, test_labellines)
        print("数据集处理完成。")
        print("加载预训练词向量。。。")
        # glove_file = datapath('data/glove.6B.300d.txt')
        tmp_file = get_tmpfile("C:\\Users\\dell\\Desktop\\textcnn\\data\\word2vec.txt")
        # glove2word2vec(glove_file, tmp_file)
        wvmodel = KeyedVectors.load_word2vec_format(tmp_file)

        wordvector = torch.zeros(args.vocab_size + 1, args.embedding_size)
        # 遍历预训练的词嵌入的 400000个词
        for i in range(len(wvmodel.index2word)):
            index = word2idx[wvmodel.index2word[i]]
            # 不为 [UNK]
            if index != 1:
                wordvector[index, :] = torch.from_numpy(wvmodel.get_vector(idx2word[word2idx[wvmodel.index2word[i]]]))

        lossword_num = -2
        flag = torch.zeros(args.embedding_size)
        for i in range(len(wordvector)):
            if wordvector[i].equal(flag):
                lossword_num += 1
                wordvector[i] = torch.from_numpy(np.random.uniform(-0.1, 0.1, [args.embedding_size])).type(torch.FloatTensor)
        print(lossword_num, "个词无预训练的词嵌入，随机初始化")
        wordvector[0] = torch.zeros(args.embedding_size).type(torch.FloatTensor)
        print("预训练词向量加载完成。")

        with open('data/mapping.pkl', 'wb') as f:
            mappings = {
                # 'word2idx': word2idx,
                # 'idx2word': idx2word,
                'train_dataset': train_dataset,
                'test_dataset': test_dataset,
                'wordvector': wordvector
            }
            cPickle.dump(mappings, f)
        print("存储完成。")