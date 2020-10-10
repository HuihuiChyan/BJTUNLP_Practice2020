#数据处理
import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

root = '/data/yinli/dataset/task1/'
class_name = ['neg', 'pos']

#替换缩略词
def replace_abbreviations(text):
    text = text.lower().replace("it's", "it is").replace("i'm", "i am").replace("he's", "he is").replace("she's", "she is")\
            .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's", "that is")\
            .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't", "does not")\
            .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not")\
            .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's", "let us")\
            .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't", "could not")\
            .replace("you'll", "you will").replace("you've", "you have")
    return text

#去除所有标点符号
def clear_review(text):
    text = text.replace("<br /><br />", "")
    text = re.sub("[^a-zA-Z]", " ", text.lower())
    return text

#去停用词，并将动词变回原形
def stemed_words(text):
    stop_words = stopwords.words("english")
    lemma = WordNetLemmatizer()
    words = [lemma.lemmatize(w, pos='v') for w in text.split() if w not in stop_words]
    result = " ".join(words)
    return result

def process(text):
    text = replace_abbreviations(text)
    text = clear_review(text)
    text = stemed_words(text)
    return text

#处理训练数据
train_path = 'train.txt'
file = open(os.path.join(root, train_path), 'w+', encoding='utf-8')
for name in class_name:
    file_name_list = os.listdir(os.path.join(root, 'train', name))
    for f_name in file_name_list:
        with open(os.path.join(root, 'train', name, f_name), encoding='utf-8') as f:
            line = f.read()
            line = process(line)
            file.write(name + ' ' + line + '\n')
file.close()

#处理验证数据
dev_path = 'dev.txt'
file = open(os.path.join(root, dev_path), 'w+', encoding='utf-8')
for name in class_name:
    file_name_list = os.listdir(os.path.join(root, 'test', name))
    for f_name in file_name_list:
        with open(os.path.join(root, 'test', name, f_name), encoding='utf-8') as f:
            line = f.read()
            line = process(line)
            file.write(name + ' ' + line + '\n')
file.close()

#处理测试数据
test_path = 'test.txt'
file = open(os.path.join(root, test_path), 'w+', encoding='utf-8')
with open(os.path.join(root, 'test_raw.txt'), encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        line = process(line)
        file.write(line + '\n')
file.close()