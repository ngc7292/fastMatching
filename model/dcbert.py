# -*- coding: utf-8 -*-
import torch
from torch import nn

from fastNLP.embeddings import BertEmbedding
from torch.nn import Transformer
from fastNLP.core.utils import seq_len_to_mask


class MyDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p

    def forward(self, x):
        if self.training and self.p > 0.0001:
            # print('mydropout!')
            mask = torch.rand(x.size())
            # print(mask.device)
            mask = mask.to(x)
            # print(mask.device)
            mask = mask.lt(self.p)
            x = x.masked_fill(mask, 0) / (1 - self.p)
        return x


class BERT_Matching(nn.Module):
    def __init__(self, dataBundle, args):
        super().__init__()
        self.vocabs = dataBundle.vocabs
        self.ptm_encoder = BertEmbedding(
            self.vocabs['words1'],
            model_dir_or_name=args.model_dir_or_name,
            pool_method=args.pool_method,
            include_cls_sep=True,
            auto_truncate=True
        )

        self.bert_dropout = MyDropout(args.bert_dropout)

        self.loss_func = nn.CrossEntropyLoss()
        self.linear_after_ptm = nn.Linear(self.ptm_encoder.embed_size, args.transformer_dim)
        self.transformer = Transformer(
            num_encoder_layers=args.transformer_layer,
            num_decoder_layers=args.transformer_layer,
            dim_feedforward=self.ptm_encoder.embed_size,
            d_model=args.transformer_dim,
            nhead=args.transformer_num_head,
            dropout=args.transformer_dropout
        )

        self.decoder_input_dim = args.transformer_dim * 2
        if args.decoder_mode == "linnear":
            self.ffn = nn.Linear(self.decoder_input_dim, len(self.vocabs['target']))
        elif args.decoder_mode == "multi":
            self.ffn = nn.Sequential(nn.Linear(self.decoder_input_dim, self.decoder_input_dim * 2),
                                     nn.LeakyReLU(), MyDropout(args.decode_dropout),
                                     nn.Linear(self.decoder_input_dim * 2, len(self.vocabs['target'])))
        else:
            raise NotImplementedError

    def forward(self, **kwargs):
        seq_len1 = kwargs['seq_len1']
        seq_len2 = kwargs['seq_len2']
        target = kwargs['target']

        words1 = kwargs['words1']
        words2 = kwargs['words2']
        batch_size = words1.size(0)

        # pre train encoder
        encoded_1 = self.encode_words(words1)
        encoded_2 = self.encode_words(words2)

        # decoder
        pred = self.transformer_and_decoder(encoded_1, encoded_2, batch_size, seq_len1, seq_len2)

        loss = self.loss_func(pred, target)
        result = {'pred': pred, 'loss': loss}

        return result

    def encode_words(self, words):
        return self.bert_dropout(self.ptm_encoder(words))

    def transformer_and_decoder(self, encoded_1, encoded_2, batch_size, seq_len1, seq_len2):
        seq_len1 += 2  # add [cls], [seq]
        seq_len2 += 2  # add [cls], [seq]

        encoded_cat = self.linear_after_ptm(torch.cat([encoded_1, encoded_2], dim=1))

        encoded_1_mask = seq_len_to_mask(seq_len1, max_len=encoded_1.size(1))
        encoded_2_mask = seq_len_to_mask(seq_len2, max_len=encoded_2.size(1))

        encoded_concat_mask = torch.cat([encoded_1_mask, encoded_2_mask], dim=1)

        transformer_out = self.transformer(encoded_cat, encoded_concat_mask)

        # 取出 transformer cls 拼在一起过decoder
        cls_1_2 = transformer_out[:, [0, encoded_1.size(1)]]
        cls_1_2 = cls_1_2.view(batch_size, -1)

        pred = self.decoder(cls_1_2)
        return pred

    def get_encoder_token(self, words):
        return self.encode_words(words)

    def predict(self):
        pass
