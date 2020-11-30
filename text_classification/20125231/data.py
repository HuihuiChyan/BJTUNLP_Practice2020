import glob
import pdb

train_negs = glob.glob(r'./aclImdb/train/neg/*')
train_poss = glob.glob(r'./aclImdb/train/pos/*')
test_negs = glob.glob(r'./aclImdb/test/neg/*')
test_poss = glob.glob(r'./aclImdb/test/pos/*')
train_neg_lines = []
train_pos_lines = []
test_neg_lines = []
test_pos_lines = []


for file in train_negs:
    with open(file,'r',encoding='utf-8') as fin:
        line = fin.readlines()[0].strip()
        train_neg_lines.append(line)
with open('train_negs.txt','w',encoding='utf-8') as fout:
    for line in train_neg_lines:
        fout.write(line+'\n')

for file in train_poss:
    with open(file,'r',encoding='utf-8') as fin:
        line = fin.readlines()[0].strip()
        train_pos_lines.append(line)
with open('train_poss.txt','w',encoding='utf-8') as fout:
    for line in train_pos_lines:
        fout.write(line+'\n')

for file in test_negs:
    with open(file,'r',encoding='utf-8') as fin:
        line = fin.readlines()[0].strip()
        test_neg_lines.append(line)
with open('test_negs.txt','w',encoding='utf-8') as fout:
    for line in test_neg_lines:
        fout.write(line+'\n')

for file in test_poss:
    with open(file,'r',encoding='utf-8') as fin:
        line = fin.readlines()[0].strip()
        test_pos_lines.append(line)
with open('test_poss.txt','w',encoding='utf-8') as fout:
    for line in test_pos_lines:
        fout.write(line+'\n')