# -*- coding: utf-8 -*-
"""
__title__="bert_for_qa"
__author__="ngc7293"
__mtime__="2021/1/19"
"""
import torch
import torch.functional as F
from torch import nn
from fastNLP.models import BaseModel
from fastNLP.embeddings import BertEmbedding


class BertForQuestionAnswering(BaseModel):
    r"""
    用于做Q&A的Bert模型，如果是Squad2.0请将BertEmbedding的include_cls_sep设置为True，Squad1.0或CMRC则设置为False

    """

    def __init__(self, bundle, args):
        super(BertForQuestionAnswering, self).__init__()
        self.vocab = bundle.vocabs["words"]

        self.bert = BertEmbedding(vocab=bundle.vocabs['words'],
                                  model_dir_or_name=args.model_dir_or_name,
                                  layers=args.layers,
                                  pool_method=args.pool_method,
                                  dropout=args.bert_dropout,
                                  include_cls_sep=False,
                                  auto_truncate=True)
        self.qa_outputs = nn.Linear(self.bert.embedding_dim, 2)

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, **kwargs):
        """
        输入words为question + [SEP] + [paragraph]，BERTEmbedding在之后会额外加入开头的[CLS]和结尾的[SEP]. note:
            如果BERTEmbedding中include_cls_sep=True，则输出的start和end index相对输入words会增加一位；如果为BERTEmbedding中
            include_cls_sep=False, 则输出start和end index的位置与输入words的顺序完全一致
        :param kwargs:
        :return:
        """
        words = kwargs['words']
        seq_len1 = kwargs['seq_len1']
        answer_start = kwargs['answer_start'] + seq_len1 + 1
        answer_end = kwargs['answer_end'] + seq_len1 + 1

        sequence_output = self.bert(words)
        logits = self.qa_outputs(sequence_output)  # [batch_size, seq_len, num_labels]

        pred_start = logits[:, :, 0]
        pred_end = logits[:, :, 1]

        start_loss = self.loss_func(pred_start, answer_start)
        end_loss = self.loss_func(pred_end, answer_end)

        total_loss = (start_loss + end_loss) / 2

        return {'pred_start': pred_start, 'pred_end': pred_end, 'loss': total_loss}

    def predict(self, **kwargs):
        """
        return pred_answer
        :param kwargs:
        :return:
        """
        words = kwargs['words'].detach().tolist()
        target_text = kwargs['answer']
        forward = self.forward(**kwargs)
        pred_start = torch.argmax(forward['pred_start'], dim=-1).detach().tolist()
        pred_end = torch.argmax(forward['pred_end'], dim=-1).detach().tolist()

        pred_answer = []
        for word, start, end in zip(words, pred_start, pred_end):
            sentence = []
            for i in word[start:end]:
                sentence.append(self.vocab.to_word(i))
            pred_answer.append(" ".join(sentence))

        return {'pred': pred_answer, 'target': target_text}
