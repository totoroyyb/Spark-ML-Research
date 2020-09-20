# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:49:30 2020

@author: yibo yan
"""

import os
import argparse
import numpy as np

vocab_size = "--vocab_size"
sentence_length = "--sentence_length"
hidden_size = "--hidden_size"
embedding_size = "--embedding_size"
dropout_rate = "--dropout_rate"
learning_rate = "--learning_rate"
epoch = "--epoch"
batch_size = "--batch_size"

file_name = "fake_news_lstm.py"


def run_vocab_size_test(init, stride, rep):
    init = int(init)
    for i in range(init, init + int(stride) * rep, int(stride)):
        os.system("python {} {} {} --name {}".format(file_name, vocab_size, i, "vocab_size-{}".format(i)))
    
def run_sentence_length_test(init, stride, rep):
    init = int(init)
    for i in range(init, init + int(stride) * rep, int(stride)):
        os.system("python {} {} {} --name {}".format(file_name, sentence_length, i, "sentence_length-{}".format(i)))
        
def run_hidden_size_test(init, stride, rep):
    init = int(init)
    for i in range(rep):
        size = init * (int(stride) ** i)
        os.system("python {} {} {} --name {}".format(file_name, hidden_size, size, "hidden_size-{}".format(size)))
        
def run_embedding_size_test(init, stride, rep):
    init = int(init)
    for i in range(rep):
        size = init * (int(stride) ** i)
        os.system("python {} {} {} --name {}".format(file_name, embedding_size, size, "embedding_size-{}".format(size)))
        
def run_dropout_rate_test(init, stride, rep):
    assert init + rep * stride <= 1, "Dropout rate can be larger than 1 at the end."
    for i in np.linspace(init, init + rep * stride, rep + 1):
        os.system("python {} {} {} --name {}".format(file_name, dropout_rate, i, "dropout_rate-{:.2f}".format(i)))
        
def run_learning_rate_test(init, stride, rep):
    for i in range(rep):
        size = init * (int(stride) ** i)
        os.system("python {} {} {} --name {}".format(file_name, learning_rate, size, "learning_rate-{}".format(size)))
        
def run_batch_size_test(init, stride, rep):
    init = int(init)
    for i in range(rep):
        size = init * (int(stride) ** i)
        os.system("python {} {} {} --name {}".format(file_name, batch_size, size, "batch_size-{}".format(size)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variable', default='epoch', type=str,
                        help="Indicate which variable need to be tested.")
    parser.add_argument('--init', default=5, type=float,
                        help="The initial value for testing.")
    parser.add_argument('--stride', default=5, type=float,
                        help="The interval of each test.")
    parser.add_argument('--rep', default=20, type=int,
                        help="The number of times need to be tested.")
    args = parser.parse_args()
    
    variable = args.variable
    
    if variable == "vocab_size":
        run_vocab_size_test(args.init, args.stride, args.rep)
    elif variable == "sentence_length":
        run_sentence_length_test(args.init, args.stride, args.rep)
    elif variable == "hidden_size":
        run_hidden_size_test(args.init, args.stride, args.rep)
    elif variable == "embedding_size":
        run_embedding_size_test(args.init, args.stride, args.rep)
    elif variable == "dropout_rate":
        run_dropout_rate_test(args.init, args.stride, args.rep)
    elif variable == "learning_rate":
        run_learning_rate_test(args.init, args.stride, args.rep)
    elif variable == "batch_size":
        run_batch_size_test(args.init, args.stride, args.rep)
    
