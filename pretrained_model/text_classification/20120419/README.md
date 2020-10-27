    
    预训练模型：使用英文的bert-base-uncased
    超参:
              batch_size: 16
              lr: 2e-5
              max_len: 256
    剩余部分参考论文： 《How to Fine-Tune BERT for Text Classification?》
    对于较长句子的处理：由于512句长的模型放不下，所以使用256最大长度，对于长度超过256的长文本，截取文本前128和后128作为新的文本【ps:是添加[CLS]和[SEP]后截取】
    后期优化点：①将12层的encoder采取梯度下降的lr，比例为0.95 
                       ②对预训练模型采取ITPT方式进行further_training，使用25000条训练文本进行额外的预训练。