#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, kernel_size, char_embedding_size, filter_size, max_word_length):
        super(CNN, self).__init__()
        self.k = kernel_size
        self.char_embedding_size = char_embedding_size
        self.filter_size = filter_size
        self.max_word_length = max_word_length
        self.cnn = nn.Conv1d(self.char_embedding_size, self.filter_size, self.k, bias=True)
        self.pool = nn.MaxPool1d(self.max_word_length - self.k + 1)

    def forward(self, x_char_emb):
        return self.pool(self.cnn(x_char_emb))

### END YOUR CODE

