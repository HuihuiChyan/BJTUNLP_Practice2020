论文阅读顺序：
《Attention is All You Need》
《Deep Contextualized Word Representations》
《Improving Language Understanding by Generative Pre-Training》
《BERT- Pre-training of Deep Bidirectional Transformers for Language Understanding》

----

## 1 EMLo

Introduction：传统的词向量无法对一词多义进行建模，Word Embedding本质上是个静态的方式，训练好之后每个单词的表达就固定住了

ELMo则是事先用语言模型学好一个单词的Word Embedding，再结合biLM得到的正向输出和反向输出，根据权重得到新的Word Embedding

ELMo Method: use bidirectional LSTM

缺点：ELMo本质还是一个模型，不同的任务训练得到的词向量不同，在实际使用时根据上下文信息去调整word embedding，解决多义词问题

ELMo input：ELMo的输入并不是one-hot编码，而是cnn+big+lstm，记作xk

目标函数：jointly maximizes the log likelihood of the forward and backward directions

each token tk, a L-layer biLM computes a set of 2L + 1 representations：L个前向的输出向量，L个后向的输出向量，1个自己本身的embedding（也就是ELMo的输入）xk

----

## 2 GPT

Two stage：
1、利用语言模型进行预训练
2、通过Fine-tuning的模式解决下游任务

与EMLo的不同之处：
1、特征抽取器不是用的RNN，而是用的Transformer
2、采用单向的语言模型，只看上文来进行预测，而抛开了下文；而EMLo看上下文

GPT需要改造下游任务，使得任务的网络结构和GPT的网络结构是一样的。然后利用第一步预训练好的参数初始化GPT的网络结构，再次，用任务去训练这个网络，对网络参数进行Fine-tuning
Classification：加上开始和结束符
Entailment：加上开始和结束符，前提和假设加上分隔符
Similarity：加上开始和结束符，两个句子之间加上分隔符，进行Transformer；接着把两个句子顺序颠倒，加上开始、结束符和分隔符，再进行Transformer
Multiple Choice同理

----

3 Bert

Bert和GPT
相同：先语言模型预训练，再使用Fine-Tuning模式解决下游任务
不同：1、Bert是双向的，看上下文信息；而GPT只看上文信息，单向的
      2、Bert基于Transformer的encoder部分架构，而GPT基于decoder架构

Bert改造下游任务：
句子关系类如Entailment：「CLS」对应的Transformer的最后一层 + Softmax
分类问题如Classification：「CLS」对应的Transformer的最后一层 + Softmax
序列标注问题：输出的每个单词对应的Transformer的最后一层 + Softmax


----

反正就是调包，关键还是理解Transformer和Bert的发展吧。Transformer的笔记有点点乱了，就不放了。