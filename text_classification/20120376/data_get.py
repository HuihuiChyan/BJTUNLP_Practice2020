from sacremoses import MosesTokenizer

f_test = open('test_split.txt', 'w')

with open('./test.txt', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]

#转换大小写开分词存入列表
mt = MosesTokenizer(lang='en')
test_feature_line = [mt.tokenize(line.lower(), return_str=True) for line in test_feature_line]

for line in test_feature_line:
    f_test.write(line+'\n')


exit()

f_train = open('train.txt', 'w')
f_valid = open('valid.txt', 'w')

#读入训练集
with open('./train_pos.txt', encoding='utf-8') as ftrain_feature:
    train_feature_line = [line.strip() for line in ftrain_feature.readlines()]
with open('./train_neg.txt', encoding='utf-8') as ftrain_feature:
    temp = [line.strip() for line in ftrain_feature.readlines()]
train_feature_line.extend(temp)

#读入验证集
with open('./valid_pos.txt', encoding='utf-8') as ftest_feature:
    test_feature_line = [line.strip() for line in ftest_feature.readlines()]
with open('./valid_neg.txt', encoding='utf-8') as ftest_feature:
    temp = [line.strip() for line in ftest_feature.readlines()]
test_feature_line.extend(temp)

#转换大小写开分词存入列表
mt = MosesTokenizer(lang='en')
train_feature_line = [mt.tokenize(line.lower(), return_str=True) for line in train_feature_line]
test_feature_line = [mt.tokenize(line.lower(), return_str=True) for line in test_feature_line]

for line in train_feature_line:
    f_train.write(line+'\n')
for line in test_feature_line:
    f_valid.write(line+'\n')

f_train.close()
f_valid.close()

