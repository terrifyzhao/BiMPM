import os
import pandas as pd
import args
import numpy as np
from gensim.models import Word2Vec
import jieba
from data_utils import pad_sequences, shuffle
import tensorflow as tf

model = Word2Vec.load('w2v/word2vec.model')


def load_char_vocab():
    vocab = [str(line).split()[0] for line in open('input/char_vocab.txt', encoding='utf-8').readlines()]
    word2idx = {word: index for index, word in enumerate(vocab)}
    idx2word = {index: word for index, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(p_sentences, h_sentences):
    word2idx, idx2word = load_char_vocab()

    p_list, h_list = [], []
    for p_sentence, h_sentence in zip(p_sentences, h_sentences):
        p = [word2idx[word] for word in p_sentence if len(word.strip()) > 0]
        h = [word2idx[word] for word in h_sentence if len(word.strip()) > 0]

        p_list.append(p)
        h_list.append(h)

    p_list = pad_sequences(p_list, maxlen=args.max_char_len)
    h_list = pad_sequences(h_list, maxlen=args.max_char_len)

    return p_list, h_list


def w2v(word):
    return model.wv[word]


def w2v_process(vec):
    if len(vec) > args.max_word_len:
        vec = vec[0:args.max_word_len]
    elif len(vec) < args.max_word_len:
        zero = np.zeros(args.word_embedding_len)
        length = args.max_word_len - len(vec)
        for i in range(length):
            vec = np.vstack((vec, zero))
    return vec


def load_data(file):
    path = os.path.join(os.path.dirname(__file__), file)
    df = pd.read_csv(path)
    p = df['sentence1'].values
    h = df['sentence2'].values
    label = df['label'].values

    p, h, label = shuffle(p, h, label)

    p_index, h_index = create_data(p, h)

    p_seg = map(lambda x: list(jieba.cut(x)), p)
    h_seg = map(lambda x: list(jieba.cut(x)), h)

    p_vec = map(lambda x: w2v(x), p_seg)
    h_vec = map(lambda x: w2v(x), h_seg)

    p_vec = np.array(list(map(lambda x: w2v_process(x), p_vec)))
    h_vec = np.array(list(map(lambda x: w2v_process(x), h_vec)))

    return p_index, h_index, p_vec, h_vec, label


def batch_data(file, batch_size):
    p_index, h_index, p_vec, h_vec, label = load_data(file)
    p_index_holder = tf.placeholder(dtype=tf.int32, shape=[None, args.max_char_len], name='p_index')
    h_index_holder = tf.placeholder(dtype=tf.int32, shape=[None, args.max_char_len], name='h_index')
    p_vec_holder = tf.placeholder(dtype=tf.float32, shape=[None, args.max_word_len, args.char_embedding_len],
                                  name='p_vec')
    h_vec_holder = tf.placeholder(dtype=tf.float32, shape=[None, args.max_word_len, args.char_embedding_len],
                                  name='h_vec')
    dataset = tf.data.Dataset.from_tensor_slices([p_index_holder, h_index_holder, p_vec_holder, h_vec_holder])
    dataset.batch(batch_size).repeat(args.epochs)

    iterator = dataset.make_initializable_iterator()
    element = iterator.get_next()


def load_fake_data():
    p, h, label = [], [], []
    for i in range(10):
        p.append(np.arange(10))
        h.append(np.arange(10))
        label.append(1)
    return p, h, label


if __name__ == '__main__':
    load_data('input/dev.csv')
