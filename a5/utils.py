#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
nmt.py: NMT Model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
"""

import math
from typing import List

import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce

def pad_sents_char(sents, char_pad_token):
    """ Pad list of sentences according to the longest sentence in the batch and max_word_length.
    @param sents (list[list[list[int]]]): list of sentences, result of `words2charindices()`
        from `vocab.py`
    @param char_pad_token (int): index of the character-padding token
    @returns sents_padded (list[list[list[int]]]): list of sentences where sentences/words shorter
        than the max length sentence/word are padded out with the appropriate pad token, such that
        each sentence in the batch now has same number of words and each word has an equal
        number of characters
        Output shape: (batch_size, max_sentence_length, max_word_length)
    """
    # Words longer than 21 characters should be truncated
    max_word_length = 21
    max_sens_length = 0

    ### YOUR CODE HERE for part 1f
    ### TODO:
    ###     Perform necessary padding to the sentences in the batch similar to the pad_sents()
    ###     method below using the padding character from the arguments. You should ensure all
    ###     sentences have the same number of words and each word has the same number of
    ###     characters.
    ###     Set padding words to a `max_word_length` sized vector of padding characters.
    ###
    ###     You should NOT use the method `pad_sents()` below because of the way it handles
    ###     padding and unknown words.
    sents_padded = deepcopy(sents)

    padded_word = [char_pad_token for _ in range(max_word_length)]
    for sentence in sents_padded:
        sentence_length = len(sentence)
        if(sentence_length > max_sens_length):
            max_sens_length = sentence_length
        for word in sentence:
            word_length = len(word)
            for _ in range(word_length, max_word_length):
                word.append(char_pad_token)

    for sentence in sents_padded:
        sentence_length = len(sentence)
        for _ in range(sentence_length, max_sens_length):
            sentence.append(padded_word)

    ### END YOUR CODE

    return sents_padded


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[int]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (int): padding token
    @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
        Output shape: (batch_size, max_sentence_length)
    """
    sents_padded = []

    ### COPY OVER YOUR CODE FROM ASSIGNMENT 4
    max_length = reduce(max, list(map(len, sents)))
    sents_padded = list(map(lambda x: x + [pad_token for i_ in range(max_length - len(x))], sents))

    ### END YOUR CODE FROM ASSIGNMENT 4

    return sents_padded



def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents
