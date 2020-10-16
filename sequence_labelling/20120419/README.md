词向量直接使用glove_6b_200.txt，后期有待调整，
具体做法和论文相同，通过word+char向量拼接作为Bi-LSTM的input，Bi-LSTM的输出接CRF层进行预测。
本地调试发现200d比100d效果稍好一点，可能部分参数没有调优导致的。