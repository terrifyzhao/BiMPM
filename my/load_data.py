import torch.nn as nn
import torch
from my import model_constant
import os
import pandas as pd
import jieba
from my import args
import numpy as np


def load_char_vocab():
    vocab = [str(line).split()[0] for line in open('input/char_vocab.txt').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence]
        h = [word2idx[word] for word in h_sentence]
        if len(p) > args.max_char_len:
            p_list.append(np.array(p[0:args.max_char_len]))
        else:
            p_list.append(np.array(p))

        if len(h) > args.max_char_len:
            h_list.append(np.array(h[0:args.max_char_len]))
        else:
            h_list.append(np.array(h))

    P = np.zeros([len(p_list), args.max_char_len], np.int)
    H = np.zeros([len(h_list), args.max_char_len], np.int)

    for i, (p, h) in enumerate(zip(p_list, h_list)):
        P[i] = np.lib.pad(p, [0, args.max_char_len - len(p)], 'constant', constant_values=(0, 0))
        H[i] = np.lib.pad(h, [0, args.max_char_len - len(h)], 'constant', constant_values=(0, 0))

    return P, H


def load_data(file):
    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values[0:args.batch_size]
    h = df['sentence2'].values[0:args.batch_size]
    label = df['label'].values[0:args.batch_size]

    p = list(map(lambda x: [i for i in x if len(i.strip()) > 0], p))
    h = list(map(lambda x: [i for i in x if len(i.strip()) > 0], h))

    p, h = create_data(p, h)

    return p, h, label


def load_fake_data():
    p, h, label = [], [], []
    for i in range(10):
        p.append(np.arange(10))
        h.append(np.arange(10))
        label.append(1)
    return p, h, label


# if __name__ == '__main__':
#     p, h, label = load_data('dev.csv')
