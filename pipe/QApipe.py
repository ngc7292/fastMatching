# -*- coding: utf-8 -*-
"""
__title__="QApipe"
__author__="ngc7293"
__mtime__="2021/1/19"
"""
import warnings

from fastNLP.io.pipe.utils import get_tokenizer
from fastNLP.io.pipe import Pipe, QNLIBertPipe
from fastNLP.io import DataBundle
from fastNLP.core.const import Const
from fastNLP.core.vocabulary import Vocabulary

from loader.QAloader import SQuADLoader

QATARGET = 'answer'
QATARGET1 = 'answer_start'
QATARGET2 = 'answer_end'


# concat两个words
def concat(ins):
    words0 = ins[Const.INPUTS(0)]
    words1 = ins[Const.INPUTS(1)]
    words = words0 + ['[SEP]'] + words1
    return words



def get_answer_end(ins):
    return ins[QATARGET1] + len(ins[Const.INPUTS(1)])


class QAPipe(Pipe):
    """
    this pipe is for question answering task
    """

    def __init__(self, lower=False, tokenizer='raw'):
        super().__init__()
        self.lower = bool(lower)
        self.tokenizer = get_tokenizer(tokenize_method=tokenizer)

    def _tokenize(self, data_bundle, field_names, new_field_names):
        """
        :param ~fastNLP.DataBundle data_bundle: DataBundle.
        :param list field_names: List[str], 需要tokenize的field名称
        :param list new_field_names: List[str], tokenize之后field的名称，与field_names一一对应。
        :return: 输入的DataBundle对象
        """
        for name, dataset in data_bundle.datasets.items():
            for field_name, new_field_name in zip(field_names, new_field_names):
                dataset.apply_field(lambda words: self.tokenizer(words), field_name=field_name,
                                    new_field_name=new_field_name)
        return data_bundle

    def process(self, data_bundle):
        r"""
        接受的DataBundle中的DataSet应该具有以下的field, target列可以没有

        .. csv-table::
           :header: "raw_words1", "raw_words2","answer_start", "answer_end", "answer_text"

        :param ~fastNLP.DataBundle data_bundle: 通过loader读取得到的data_bundle，里面包含了数据集的原始数据内容
        :return: data_bundle
        """
        data_bundle = self._tokenize(data_bundle, [Const.RAW_WORDS(0), Const.RAW_WORDS(1)],
                                     [Const.INPUTS(0), Const.INPUTS(1)])

        if self.lower:
            for name, dataset in data_bundle.datasets.items():
                dataset[Const.INPUTS(0)].lower()
                dataset[Const.INPUTS(1)].lower()

        for name, dataset in data_bundle.datasets.items():
            dataset.apply(concat, new_field_name=Const.INPUT)
            if dataset.has_field(QATARGET1):
                dataset.drop(lambda x: x[QATARGET1] == '-')
            if dataset.has_field(QATARGET2):
                dataset.drop(lambda x: x[QATARGET2] == '-')

        word_vocab = Vocabulary()
        word_vocab.from_dataset(*[dataset for name, dataset in data_bundle.datasets.items() if 'train' in name],
                                field_name=Const.INPUT,
                                no_create_entry_dataset=[dataset for name, dataset in data_bundle.datasets.items() if
                                                         'train' not in name])
        word_vocab.index_dataset(*data_bundle.datasets.values(),
                                 field_name=[Const.INPUT, Const.INPUTS(0), Const.INPUTS(1)])

        data_bundle.set_vocab(word_vocab, Const.INPUT)

        input_fields = ['id', Const.INPUT, Const.INPUTS(0), Const.INPUTS(1), Const.INPUT_LEN, Const.INPUT_LENS(0),
                        Const.INPUT_LENS(1), QATARGET, QATARGET1, QATARGET2]
        target_fields = [QATARGET, QATARGET1, QATARGET2]

        for name, dataset in data_bundle.datasets.items():
            dataset.add_seq_len(Const.INPUT)
            dataset.add_seq_len(Const.INPUTS(0), new_field_name=Const.INPUT_LENS(0))
            dataset.add_seq_len(Const.INPUTS(1), new_field_name=Const.INPUT_LENS(1))
            dataset.set_input(*input_fields, flag=True)
            for fields in target_fields:
                if dataset.has_field(fields):
                    dataset.set_target(fields, flag=True)

        return data_bundle

    def process_from_file(self, paths) -> DataBundle:
        return self.process(SQuADLoader().load(paths))
