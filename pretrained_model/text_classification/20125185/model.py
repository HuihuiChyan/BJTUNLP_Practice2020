#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import BertTokenizer, BertModel,AdamW
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
import time


class Bert(torch.nn.Module):
    def __init__(self):
        super(Bert,self).__init__()
        
        self.bert_model = BertModel.from_pretrained('./bert-base-uncased')
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.bert_model.config.hidden_size, 2)
    
    def forward(self, features, mask):
        output = self.bert_model(features, attention_mask=mask)
        # [batch,max_len,hidden_size]
        output = output[1]
        output = self.dropout(output)
        output = self.linear(output)
        return output

