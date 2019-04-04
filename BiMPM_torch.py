import torch
import torch.nn as nn
import torch.nn.functional as F
import args


class BiMPM(nn.Module):
    def __init__(self, word_embedding=0):
        super(BiMPM, self).__init__()

        # ----- Word Representation Layer -----
        self.char_embedding = nn.Embedding(args.char_vocab_len, args.char_embedding_len)

        # self.word_embedding = nn.Embedding(args.word_vocab_size, args.word_embedding_len)
        # self.word_embedding.weiht.data.copy_(word_embedding)
        # self.word_embedding.weight.requires = False

        self.char_LSTM = nn.LSTM(
            input_size=args.char_embedding_len,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=args.char_hidden_size,
            hidden_size=args.context_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(args.num_perspective, args.context_hidden_size)))

        # ----- Aggregation Layer -----
        self.aggregation_LSTM = nn.LSTM(
            input_size=32,
            hidden_size=args.agg_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.pred_fc1 = nn.Linear(args.agg_hidden_size * 4, args.agg_hidden_size * 2)
        self.pred_fc2 = nn.Linear(args.agg_hidden_size * 2, args.class_size)

    def forward(self, **kwargs):

        def matching_func_full(v1, v2, w):
            # 转置，添加维度
            w = torch.transpose(w, 1, 0).unsqueeze(0).unsqueeze(0)
            # 把v1在dim维度上扩充args.num_perspective倍
            v1 = w * torch.stack([v1] * args.num_perspective, dim=3)
            if len(v2.size()) == 3:
                v2 = w * torch.stack([v2] * args.num_perspective, dim=3)
            else:
                v2 = w * torch.stack([torch.stack([v2] * v1.size(1), dim=1)] * args.num_perspective, dim=3)

            m = F.cosine_similarity(v1, v2, dim=2)
            return m

        def div_with_small_value(n, d, eps=1e-8):
            # too small values are replaced by 1e-8 to prevent it from exploding.
            d = d * (d > eps).float() + eps * (d <= eps).float()
            return n / d

        def matching_func_maxpool(v1, v2, w):
            w = w.unsqueeze(0).unsqueeze(2)
            v1, v2 = w * torch.stack([v1] * args.num_perspective, dim=1), \
                     w * torch.stack([v2] * args.num_perspective, dim=1)

            v1_norm = v1.norm(p=2, dim=3, keepdim=True)
            v2_norm = v2.norm(p=2, dim=3, keepdim=True)

            n = torch.matmul(v1, v2.transpose(2, 3))
            d = v1_norm * v2_norm.transpose(2, 3)

            m = div_with_small_value(n, d).permute(0, 2, 3, 1)

            return m

        def attention(v1, v2):
            v1_norm = v1.norm(p=2, dim=2, keepdim=True)
            v2_norm = v2.norm(p=2, dim=2, keepdim=True).permute(0, 2, 1)

            a = torch.bmm(v1, v2.permute(0, 2, 1))
            d = v1_norm * v2_norm

            return div_with_small_value(a, d)

        # ----- Word Representation Layer -----
        # (batch, seq_len, max_word_len)
        seq_len_p = kwargs['p_char'].size(1)
        seq_len_h = kwargs['h_char'].size(1)

        # (batch, max_char_len, vocab_size) -> (batch*max_char_len, vocab_size)
        char_p = kwargs['p_char'].view(-1, args.max_char_len)
        char_h = kwargs['h_char'].view(-1, args.max_char_len)

        # (batch*max_char_len, vocab_size) -> (batch*max_char_len, vocab_size, char_embedding_len)
        # (batch*max_char_len, char_embedding_len) -> (batch, max_char_len, char_hidden_size)
        _, (char_p_embedding, _) = self.char_LSTM(self.char_embedding(char_p))
        _, (char_h_embedding, _) = self.char_LSTM(self.char_embedding(char_h))

        # (1, batch*seq_len, char_hidden_size) -> (batch, seq_len, char_hidden_size)
        char_p_embedding = char_p_embedding.view(args.batch_size, -1, args.char_hidden_size)
        char_h_embedding = char_h_embedding.view(args.batch_size, -1, args.char_hidden_size)

        # (batch, seq_len) -> (batch. seq_len, word_dim)
        # word_p_embedding = self.word_embedding(kwargs['p_word'])
        # word_h_embedding = self.word_embedding(kwargs['h_word'])

        # (batch, seq_len, word_dim) -> (batch, seq_len, word_dim + char_hidden_size)
        # p = torch.cat((word_p_embedding, char_p_embedding), dim=-1)
        # h = torch.cat((word_h_embedding, char_h_embedding), dim=-1)
        # p = torch.cat((char_p_embedding, char_p_embedding), dim=-1)
        # h = torch.cat((char_h_embedding, char_h_embedding), dim=-1)

        p = F.dropout(char_p_embedding, p=args.drop_out)
        h = F.dropout(char_h_embedding, p=args.drop_out)

        # ----- Context Representation Layer -----
        # (batch, seq_len, word_dim + char_hidden_size) -> (batch, seq_len, context_hidden_size)
        p = self.context_LSTM(p)
        h = self.context_LSTM(h)
        p = self.dropout(p[0])
        h = self.dropout(h[0])

        # (batch, seq_len, hidden_size)
        p_fw, p_bw = torch.split(p, args.context_hidden_size, dim=-1)
        h_fw, h_bw = torch.split(h, args.context_hidden_size, dim=-1)

        # ----- Matching Layer -----
        # 1、Full-Matching 所有时刻的context和另一个序列的序列context计算相似度
        # (batch, seq_len, 1)
        p_full_fw = matching_func_full(p_fw, h_fw[:, -1, :], self.mp_w1)
        p_full_bw = matching_func_full(p_bw, h_bw[:, 0, :], self.mp_w2)
        h_full_fw = matching_func_full(h_fw, p_fw[:, -1, :], self.mp_w1)
        h_full_bw = matching_func_full(h_bw, p_bw[:, 0, :], self.mp_w2)

        # 2、Maxpooling-Matching
        max_fw = matching_func_maxpool(p_fw, h_fw, self.mp_w3)
        max_bw = matching_func_maxpool(p_bw, h_bw, self.mp_w4)

        p_max_fw = max_fw.max(dim=2)
        p_max_bw = max_bw.max(dim=2)
        h_max_fw = max_fw.max(dim=1)
        h_max_bw = max_bw.max(dim=1)

        # 3、Attentive-Matching
        att_fw = attention(p_fw, h_fw)
        att_bw = attention(p_bw, h_bw)

        # unsqueeze增加维度
        att_h_fw = h_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_h_bw = h_bw.unsqueeze(1) * att_bw.unsqueeze(3)

        att_p_fw = p_fw.unsqueeze(1) * att_fw.unsqueeze(3)
        att_p_bw = p_bw.unsqueeze(1) * att_bw.unsqueeze(3)

        att_mean_h_fw = div_with_small_value(att_h_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_h_bw = div_with_small_value(att_h_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        att_mean_p_fw = div_with_small_value(att_p_fw.sum(dim=2), att_fw.sum(dim=2, keepdim=True))
        att_mean_p_bw = div_with_small_value(att_p_bw.sum(dim=2), att_bw.sum(dim=2, keepdim=True))

        p_att_mean_fw = matching_func_full(p_fw, att_mean_h_fw, self.mp_w5)
        p_att_mean_bw = matching_func_full(p_bw, att_mean_h_bw, self.mp_w6)
        h_att_mean_fw = matching_func_full(h_fw, att_mean_p_fw, self.mp_w5)
        h_att_mean_bw = matching_func_full(h_bw, att_mean_p_bw, self.mp_w6)

        # 4、Max-Attentive-Matching
        att_max_h_fw, _ = att_h_fw.max(dim=2)
        att_max_h_bw, _ = att_h_bw.max(dim=2)
        att_max_p_fw, _ = att_p_fw.max(dim=2)
        att_max_p_bw, _ = att_p_bw.max(dim=2)

        p_att_max_fw = matching_func_full(p_fw, att_max_h_fw, self.mp_w7)
        p_att_max_bw = matching_func_full(p_bw, att_max_h_bw, self.mp_w8)
        h_att_max_fw = matching_func_full(h_fw, att_max_p_fw, self.mp_w7)
        h_att_max_bw = matching_func_full(h_bw, att_max_p_bw, self.mp_w8)

        mv_p = torch.cat(
            (p_full_fw, p_max_fw[0], p_att_mean_fw, p_att_max_fw,
             p_full_bw, p_max_bw[0], p_att_mean_bw, p_att_max_bw), dim=2)
        mv_h = torch.cat(
            (h_full_fw, h_max_fw[0], h_att_mean_fw, h_att_max_fw,
             h_full_bw, h_max_bw[0], h_att_mean_bw, h_att_max_bw), dim=2)

        mv_p = self.dropout(mv_p)
        mv_h = self.dropout(mv_h)

        # ----- Aggregation Layer -----
        _, (agg_p_last, _) = self.aggregation_LSTM(mv_p)
        _, (agg_h_last, _) = self.aggregation_LSTM(mv_h)

        x = torch.cat(
            (agg_p_last.permute(1, 0, 2).contiguous().view(-1, args.agg_hidden_size * 2),
             agg_h_last.permute(1, 0, 2).contiguous().view(-1, args.agg_hidden_size * 2)), dim=1)
        x = self.dropout(x)

        # ----- Prediction Layer -----
        x = self.pred_fc1(x)
        x = self.dropout(x)
        x = self.pred_fc2(x)

        return x

    def dropout(self, v):
        return F.dropout(v, p=args.drop_out, training=self.training)
