#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch.nn as nn
import torch
import torch.nn.functional as F


class Highway(nn.Module):

    def __init__(self, word_embedding_size, dropout_rate=0.8):
        super(Highway, self).__init__()
        self.word_embedding_size = word_embedding_size
        self.projection = nn.Linear(self.word_embedding_size, self.word_embedding_size, bias=True)
        self.gate = nn.Linear(self.word_embedding_size, self.word_embedding_size, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, conv_out:torch.Tensor):
        x_proj = F.relu(self.projection(conv_out))
        x_gate = torch.sigmoid(self.gate(conv_out))
        x_highway = x_gate * x_proj + (1 - x_gate) * conv_out
        x_word_emb = self.dropout(x_highway)
        return x_word_emb


### END YOUR CODE 

