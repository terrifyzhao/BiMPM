import torch
import torch.nn as nn
import torch.nn.functional as F
from my import args


class BiMPM(nn.Module):
    def __init__(self, word_embedding):
        super(BiMPM, self).__init__()

        # ----- Word Representation Layer -----
        self.char_embedding = nn.Embedding(args.char_vocab_size, args.char_embedding_len)
        self.word_embedding = nn.Embedding(args.word_vocab_size, args.word_embedding_len)
        self.word_embedding.weiht.data.copy_(word_embedding)
        self.word_embedding.weight.requires = False

        self.char_LSTM = nn.LSTM(
            input_size=args.char_embedding_len,
            hidden_size=args.char_hidden_size,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )

        # ----- Context Representation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=args.char_embedding_len + args.char_hidden_size,
            hidden_size=args.context_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Matching Layer -----
        for i in range(1, 9):
            setattr(self, f'mp_w{i}', nn.Parameter(torch.rand(args.num_perspective, args.context_hidden_size)))

        # ----- Aggregation Layer -----
        self.context_LSTM = nn.LSTM(
            input_size=args.num_perspective * 8,
            hidden_size=args.context_hidden_size,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # ----- Prediction Layer -----
        self.linear1 = nn.Linear(args.context_hidden_size * 4, args.context_hidden_size * 2)
        self.linear2 = nn.Linear(args.context_hidden_size * 4, args.class_size)

    def forward(self, **kwargs):
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
        char_p_embedding = char_p_embedding.view(-1, seq_len_p, args.char_hidden_size)
        char_h_embedding = char_h_embedding.view(-1, seq_len_h, args.char_hidden_size)

        # (batch, seq_len) -> (batch. seq_len, word_dim)
        word_p_embedding = self.word_embedding(kwargs['p_char'])
        word_h_embedding = self.word_embedding(kwargs['h_char'])

        # (batch, seq_len, word_dim) -> (batch, seq_len, word_dim + char_hidden_size)
        p = torch.cat((word_p_embedding, char_p_embedding), dim=-1)
        h = torch.cat((word_h_embedding, char_h_embedding), dim=-1)

        p = F.dropout(p, p=args.drop_out)
        h = F.dropout(h, p=args.drop_out)

        # ----- Context Representation Layer -----
        # (batch, seq_len, word_dim + char_hidden_size) -> (batch, seq_len, context_hidden_size)
        p = self.context_LSTM(p)
        h = self.context_LSTM(h)
        p = self.dropout(p)
        h = self.dropout(h)

        # ----- Matching Layer -----
