from BiMPM import Graph
import tensorflow as tf
import args
from data_process import load_data
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'

p_index, h_index, p_vec, h_vec, label = load_data('input/train.csv')
p_index_dev, h_index_dev, p_vec_dev, h_vec_dev, label_dev = load_data('input/dev.csv')
p_index_holder = tf.placeholder(name='p_index', shape=(None, args.max_char_len), dtype=tf.int32)
h_index_holder = tf.placeholder(name='h_index', shape=(None, args.max_char_len), dtype=tf.int32)
p_vec_holder = tf.placeholder(name='p_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                              dtype=tf.float32)
h_vec_holder = tf.placeholder(name='h_vec', shape=(None, args.max_word_len, args.word_embedding_len),
                              dtype=tf.float32)
label_holder = tf.placeholder(name='label', shape=(None,), dtype=tf.int32)

dataset = tf.data.Dataset.from_tensor_slices((p_index_holder, h_index_holder, p_vec_holder, h_vec_holder, label_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
model = Graph()
saver = tf.train.Saver()
with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_index_holder: p_index,
                                              h_index_holder: h_index,
                                              p_vec_holder: p_vec,
                                              h_vec_holder: h_vec,
                                              label_holder: label})
    for epoch in range(args.epochs):
        step = 0
        while True:
            step += 1
            try:
                p_index_batch, h_index_batch, p_vec_batch, h_vec_batch, label_batch = sess.run(iterator.get_next())
                loss, _, predict, acc = sess.run([model.loss, model.train_op, model.predict, model.accuracy],
                                                 feed_dict={model.p: p_index_batch,
                                                            model.h: h_index_batch,
                                                            model.p_vec: p_vec_batch,
                                                            model.h_vec: h_vec_batch,
                                                            model.y: label_batch,
                                                            model.keep_prob: args.keep_prob})
                print('epoch:', epoch, ' step:', step, ' loss:', loss / args.batch_size, ' acc:', acc)
            except tf.errors.OutOfRangeError:
                print('\n')
                break

        predict, acc = sess.run([model.predict, model.accuracy],
                                feed_dict={model.p: p_index_dev,
                                           model.h: h_index_dev,
                                           model.p_vec: p_vec_dev,
                                           model.h_vec: h_vec_dev,
                                           model.y: label_dev,
                                           model.keep_prob: 1})
        print('epoch:', epoch, ' dev acc:', acc)
        saver.save(sess, f'output/BiMPM_{epoch}.ckpt')
        print('save model done')
        print('\n')

    # p, h, p_vec, h_vec, y = data_process.load_data('input/train.csv')
    # # evl_p, evl_h, evl_p_vec, evl_h_vec, evl_y = load_data.load_data('input/dev.csv')
    # model = Graph()
    # with tf.Session()as sess:
    #     sess.run(tf.global_variables_initializer())
    #     tf.summary.FileWriter('log/', sess.graph)
    #     for i in range(20):
    #         batch = int(len(y) / args.batch_size)
    #         for j in range(batch):
    #             batch_p = p[args.batch_size * j:args.batch_size * (j + 1)]
    #             batch_h = h[args.batch_size * j:args.batch_size * (j + 1)]
    #             batch_p_vec = p_vec[args.batch_size * j:args.batch_size * (j + 1)]
    #             batch_h_vec = h_vec[args.batch_size * j:args.batch_size * (j + 1)]
    #             batch_y = y[args.batch_size * j:args.batch_size * (j + 1)]
    #             loss, _, predict, acc = sess.run([model.loss, model.train_op, model.predict, model.accuracy],
    #                                              feed_dict={model.p: batch_p,
    #                                                         model.h: batch_h,
    #                                                         model.p_vec: batch_p_vec,
    #                                                         model.h_vec: batch_h_vec,
    #                                                         model.y: batch_y})
    #             print('epoch:', i, ' batch:', j, ' loss:', loss / args.batch_size, ' acc:', acc)

    # accs = []
    # for j in range(int(len(evl_y) / args.batch_size)):
    #     batch_p = evl_p[args.batch_size * j:args.batch_size * (j + 1)]
    #     batch_h = evl_h[args.batch_size * j:args.batch_size * (j + 1)]
    #     batch_p_vec = p_vec[args.batch_size * j:args.batch_size * (j + 1)]
    #     batch_h_vec = h_vec[args.batch_size * j:args.batch_size * (j + 1)]
    #     batch_y = evl_y[args.batch_size * j:args.batch_size * (j + 1)]
    # predict, acc = sess.run([model.predict, model.accuracy],
    #                         feed_dict={model.p: batch_p,
    #                                    model.h: batch_h,
    #                                    model.p_vec: batch_p_vec,
    #                                    model.h_vec: batch_h_vec,
    #                                    model.y: batch_y})
#     accs.append(acc)
# import numpy as np
# acc = np.mean(accs)
# print('evl acc: ', acc)
# print('save model')
# saver = tf.train.Saver()
# saver.save(sess, f'output/BiMPM_{i}.ckpt')
# print('')
