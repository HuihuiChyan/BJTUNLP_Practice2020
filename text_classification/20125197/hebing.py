import os
import re


def data_process(text):
    text = text.lower()
    # 特殊数据处理，该地方参考的殷同学的
    text = text.replace("<br /><br />", "").replace("it's", "it is").replace("i'm", "i am").replace("he's",
                                                                                                    "he is").replace(
        "she's", "she is") \
        .replace("we're", "we are").replace("they're", "they are").replace("you're", "you are").replace("that's",
                                                                                                        "that is") \
        .replace("this's", "this is").replace("can't", "can not").replace("don't", "do not").replace("doesn't",
                                                                                                     "does not") \
        .replace("we've", "we have").replace("i've", " i have").replace("isn't", "is not").replace("won't", "will not") \
        .replace("hasn't", "has not").replace("wasn't", "was not").replace("weren't", "were not").replace("let's",
                                                                                                          "let us") \
        .replace("didn't", "did not").replace("hadn't", "had not").replace("waht's", "what is").replace("couldn't",
                                                                                                        "could not") \
        .replace("you'll", "you will").replace("you've", "you have")

    en_regex = re.compile(r"[.…{|}#$%&\'()*+,-_./:~^;<=>@!\"]+")
    text = en_regex.sub(r'', text)
    return text


def savetxt(filetype='test'):
    classes = ['pos', 'neg']
    root_path = 'data/' + filetype
    f = open(filetype + '.txt', 'w', encoding='utf-8')
    f1 = open(filetype + '_label.txt', 'w', encoding='utf-8')
    for c in classes:
        path = os.path.join(root_path, c)
        filenames = os.listdir(path)

        for filename in filenames:
            filepath = os.path.join(path, filename)
            with open(filepath, "r", encoding='utf-8') as files:
                data = files.read()
                data = data_process(data)
                f.writelines(data)
                f.write('\n')
                f1.write(c)
                f1.write('\n')


savetxt(filetype='train')