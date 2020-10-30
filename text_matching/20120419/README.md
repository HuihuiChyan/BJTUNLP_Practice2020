就说一点:
   注意attention的计算，一开始计算attention前，直接将两个Bi-LSTM的隐藏书输出进行mask，然后进行计算socre，导致丢失了很多信息，训练过程test_acc跳动幅度很大。
   后面检查，Mask的使用，应该是在计算出score后使用与其对应的mask，来生成进一步的decoder。
