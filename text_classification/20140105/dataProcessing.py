import os
import torch
from sacremoses import MosesTokenizer

root = "./aclImdb"
label_feature = ['neg', 'pos']
mt = MosesTokenizer(lang='en')

def dataProcessing(tag):
    """
    数据处理
    """
    print('generate======>',tag)
    if tag == "train":
        w_file = open('./dataset/train.txt', 'w', encoding='utf-8')
    else:
        w_file = open('./dataset/test.txt', 'w', encoding='utf-8')
    for label in label_feature:
        file_url = os.path.join(root, tag, label)
        file_list = os.listdir(file_url)
        for name in file_list:
            with open(os.path.join(root, tag, label, name), 'r', encoding='utf-8') as file:
                line = file.read()
                # 全部转换小写 and 英文分词
                line = mt.tokenize(line.lower(), return_str=True) 
                w_file.write(label + ' ' + line+'\n')
    w_file.close()

if os.path.exists('./dataset/train.txt'):
    pass
else:
    dataProcessing('train')

if os.path.exists('./dataset/test.txt'):
    pass
else:
    dataProcessing('test')






