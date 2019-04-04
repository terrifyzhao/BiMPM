import tensorflow as tf
import args
import load_data


class Graph:
    def __init__(self):
        self.p = tf.placeholder(name='p', shape=(args.batch_size, args.max_char_len), dtype=tf.int32)
        self.h = tf.placeholder(name='h', shape=(args.batch_size, args.max_char_len), dtype=tf.int32)
        self.y = tf.placeholder(name='y', shape=(args.batch_size,), dtype=tf.int32)

        self.embed = tf.get_variable(name='embed', shape=(args.char_vocab_len, args.char_embedding_len),
                                     dtype=tf.float32)

        # self.w1 = tf.get_variable(name='w1', shape=(args.batch_size, args.char_hidden_size, args.max_char_len),
        #                           dtype=tf.float32)
        # self.w2 = tf.get_variable(name='w2', shape=(args.batch_size, args.char_hidden_size, args.max_char_len),
        #                           dtype=tf.float32)

        for i in range(1, 9):
            setattr(self, f'w{i}', tf.get_variable(name=f'w{i}', shape=(args.num_perspective, args.char_hidden_size),
                                                   dtype=tf.float32))

        self.forward()
        self.train()

    def BiLSTM(self, x):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)

        return tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, x, dtype=tf.float32)

    def LSTM(self, x):
        cell = tf.nn.rnn_cell.BasicLSTMCell(args.char_hidden_size)
        return tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    def cosine(self, v1, v2):
        v1_norm = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=2))
        v2_norm = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=2))

        cosine = tf.reduce_sum(tf.multiply(v1, v2), axis=2) / (v1_norm * v2_norm)
        return cosine

    def full_matching(self, metric, vec, w):
        w = tf.expand_dims(tf.expand_dims(tf.transpose(w), 0), 0)
        metric = w * tf.stack([metric] * args.num_perspective, axis=3)
        # vec = w * tf.stack([tf.stack([vec] * metric.shape[1], axis=1)] * args.num_perspective, axis=3)
        vec = w * tf.stack([vec] * args.num_perspective, axis=3)
        cosine = self.cosine(metric, vec)
        return cosine

    def maxpool_full_matching(self, v1, v2, w):
        w = tf.expand_dims(tf.expand_dims(tf.transpose(w), 0), 0)
        v1 = w * tf.stack([v1] * args.num_perspective, axis=3)
        v2 = w * tf.stack([v2] * args.num_perspective, axis=3)
        cosine = self.cosine(v1, v2)
        max_value = tf.reduce_max(cosine, axis=1)
        return max_value

    def forward(self):
        # ----- Word Representation Layer -----
        # 字嵌入
        p_embedding = tf.nn.embedding_lookup(self.embed, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embed, self.h)

        # 过一遍LSTM后作为字向量
        with tf.variable_scope("lstm_p", reuse=None):
            p_output, _ = self.LSTM(p_embedding)
        with tf.variable_scope("lstm_h", reuse=None):
            h_output, _ = self.LSTM(h_embedding)

        char_p_embedding = p_output[:, -1, :]
        char_h_embedding = h_output[:, -1, :]

        # ----- Context Representation Layer -----
        # 论文中是取context，tf不会输出所有时刻的ctx，这里用输出值代替
        with tf.variable_scope("bilstm_p", reuse=None):
            (p_fw, p_bw), _ = self.BiLSTM(char_p_embedding)
        with tf.variable_scope("bilstm_h", reuse=None):
            (h_fw, h_bw), _ = self.BiLSTM(char_h_embedding)

        p_fw = tf.nn.dropout(p_fw, args.drop_out)
        p_bw = tf.nn.dropout(p_bw, args.drop_out)
        h_fw = tf.nn.dropout(h_fw, args.drop_out)
        h_bw = tf.nn.dropout(h_bw, args.drop_out)

        # ----- Matching Layer -----
        p_full_fw = self.full_matching(p_fw, tf.expand_dims(h_fw[:, -1, :], 1), self.w1)
        p_full_bw = self.full_matching(p_bw, tf.expand_dims(h_bw[:, 0, :], 1), self.w2)
        h_full_fw = self.full_matching(h_fw, tf.expand_dims(p_fw[:, -1, :], 1), self.w1)
        h_full_bw = self.full_matching(h_bw, tf.expand_dims(p_bw[:, 0, :], 1), self.w2)

        # 2、Maxpooling-Matching
        max_fw = self.maxpool_full_matching(p_fw, h_fw, self.w3)
        max_bw = self.maxpool_full_matching(p_bw, h_bw, self.w4)

        # max_fw = tf.reduce_max(max_fw, axis=2)
        # max_bw = tf.reduce_max(max_bw, axis=2)
        # h_max_fw = tf.reduce_max(max_fw, axis=1)
        # h_max_bw = tf.reduce_max(max_bw, axis=1)

        # 3、Attentive-Matching
        att_fw = self.cosine(p_fw, h_fw)
        att_bw = self.cosine(p_bw, h_bw)

        att_h_fw = h_fw * tf.expand_dims(att_fw, 2)
        att_h_bw = h_bw * tf.expand_dims(att_bw, 2)
        att_p_fw = p_fw * tf.expand_dims(att_fw, 2)
        att_p_bw = p_bw * tf.expand_dims(att_bw, 2)

        att_mean_h_fw = self.cosine(att_h_fw, tf.expand_dims(att_fw, axis=2))
        att_mean_h_bw = self.cosine(att_h_bw, tf.expand_dims(att_bw, axis=2))
        att_mean_p_fw = self.cosine(att_p_fw, tf.expand_dims(att_fw, axis=2))
        att_mean_p_bw = self.cosine(att_p_bw, tf.expand_dims(att_bw, axis=2))

        p_att_mean_fw = self.maxpool_full_matching(p_fw, tf.expand_dims(att_mean_h_fw, axis=2), self.w5)
        p_att_mean_bw = self.maxpool_full_matching(p_bw, tf.expand_dims(att_mean_h_bw, axis=2), self.w6)
        h_att_mean_fw = self.maxpool_full_matching(h_fw, tf.expand_dims(att_mean_p_fw, axis=2), self.w5)
        h_att_mean_bw = self.maxpool_full_matching(h_bw, tf.expand_dims(att_mean_p_bw, axis=2), self.w6)

        # 4、Max-Attentive-Matching
        att_max_h_fw = tf.reduce_max(att_h_fw, axis=2)
        att_max_h_bw = tf.reduce_max(att_h_bw, axis=2)
        att_max_p_fw = tf.reduce_max(att_p_fw, axis=2)
        att_max_p_bw = tf.reduce_max(att_p_bw, axis=2)

        p_att_max_fw = self.maxpool_full_matching(p_fw, tf.expand_dims(att_max_h_fw, axis=2), self.w7)
        p_att_max_bw = self.maxpool_full_matching(p_bw, tf.expand_dims(att_max_h_bw, axis=2), self.w8)
        h_att_max_fw = self.maxpool_full_matching(h_fw, tf.expand_dims(att_max_p_fw, axis=2), self.w7)
        h_att_max_bw = self.maxpool_full_matching(h_bw, tf.expand_dims(att_max_p_bw, axis=2), self.w8)

        mv_p = tf.concat(
            (p_full_fw, max_fw, p_att_mean_fw, p_att_max_fw,
             p_full_bw, max_bw, p_att_mean_bw, p_att_max_bw), axis=2)

        mv_h = tf.concat(
            (h_full_fw, max_fw, h_att_mean_fw, h_att_max_fw,
             h_full_bw, max_bw, h_att_mean_bw, h_att_max_bw), axis=2)

        mv_p = tf.nn.dropout(mv_p, args.drop_out)
        mv_h = tf.nn.dropout(mv_h, args.drop_out)

        # ----- Aggregation Layer -----
        _, (agg_p_last, _) = self.BiLSTM(mv_p)
        _, (agg_h_last, _) = self.BiLSTM(mv_h)

        x = tf.concat(
            (agg_p_last.permute(1, 0, 2).contiguous().view(-1, args.agg_hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, args.agg_hidden_size * 2)), axis=1)
        x = tf.nn.dropout(x)

        # ----- Prediction Layer -----
        x = tf.layers.dense(x, 512)
        x = tf.nn.dropout(x)
        self.logits = tf.layers.dense(x, 128)

    def train(self):
        y = tf.one_hot(self.y, args.char_vocab_len)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=self.logits)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)


if __name__ == '__main__':
    p, h, y = load_data.load_fake_data()
    model = Graph()
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10):
            loss, _, _ = sess.run([model.loss, model.logits, model.train_op],
                                  feed_dict={model.p: p, model.h: h, model.y: y})
            print('loss:', loss)
