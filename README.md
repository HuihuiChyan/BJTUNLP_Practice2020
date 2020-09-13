说明：

1. 全体研一的同学，需要完成以下四个任务，每个任务大约会给一个月的时间；
2. 在模型写完并完成训练后，请联系我获取测试集，然后用自己训好的模型推断获得结果，处理成要求的格式，连同代码一并发给我；
3. 我会计算大家的测试结果，将排名公开到Github的Repository中，并一并公开代码；
4. 前三个任务可以参考开源代码，但是最后提交的代码必须是自己写的；
5. 禁止使用除了训练集以外的任何数据用于训练，禁止使用验证集、测试集进行训练；

### 任务一：基于TextCNN的文本分类

数据集：[Classify the sentiment of sentences from the Rotten Tomatoes dataset](https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews) 

参考论文：Convolutional Neural Networks for Sentence Classification，https://arxiv.org/abs/1408.5882

模型图：

![TextCNN](C:\Users\Hui_Huang\Desktop\TextCNN.PNG)

需要了解的知识点：

1. 文本特征表示：词向量
   1. 对word embedding随机初始化
   2. 用glove预训练的embedding进行初始化 https://nlp.stanford.edu/projects/glove/
2. CNN如何提取文本的特征

说明：

1. 训练集25000句，验证集25000句，需要自己写脚本合在一起；
2. 测试结果格式：每行对应一句话的分类结果；

### 任务二：基于BiLSTM+CRF的序列标注

用BiLSTM+CRF来训练序列标注模型，以Named Entity Recognition为例。

数据集：CONLL 2003，https://www.clips.uantwerpen.be/conll2003/ner/

参考论文：Neural Architectures for Named Entity Recognition，<https://arxiv.org/pdf/1603.01360.pdf> 

模型图：

![BiLSTM&CRF](C:\Users\Hui_Huang\Desktop\BiLSTM&CRF.PNG)

需要了解的知识点：

1. RNN如何提取文本的特征
2. 评价指标：precision、recall、F1
3. CRF比较复杂，不理解没关系

说明：

1. 训练集(train)、验证集(testa)、测试集(testb)已经分割好了，但是你仅使用训练集和验证集即可，最后我会再给你一个测试集；
2. 测试结果格式：每行对应一个词的标注结果，不同句之间用空行相分割；

### 任务三：基于ESIM的文本匹配

输入两个句子，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。

数据集：https://nlp.stanford.edu/projects/snli/

参考论文：Enhanced LSTM for Natural Language Inference，<https://arxiv.org/pdf/1609.06038v3.pdf>

模型图：

![ESIM](C:\Users\Hui_Huang\Desktop\ESIM.png)

知识点：

1. 注意力机制在NLP中的应用

说明：

1. 测试结果格式：每行对应一个句对的匹配结果；

### 任务四：基于Bert的自然语言理解

Bert可以用来进行分类、标注、匹配等多种自然语言理解任务。这里需要用Bert重新实现上述三个任务中的任意一个。（难度：任务一 > 任务三 > 任务二）

建议使用的框架：Huggingface，https://github.com/huggingface/transformers

参考论文：BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding，https://arxiv.org/abs/1810.04805

知识点：

1. 预训练和预训练模型
2. 子词切分
3. 自注意力和transformer（不过不需要你自己写模型）



