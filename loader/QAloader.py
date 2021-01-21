# -*- coding: utf-8 -*-
"""
__title__="QAloader"
__author__="ngc7293"
__mtime__="2021/1/19"
"""
import os
import json

from typing import Union, Dict
from tqdm import tqdm
from fastNLP.io.loader import Loader
from fastNLP.io import DataBundle
from fastNLP import DataSet, Instance


def get_word_start(context, start):
    w_list = context.split()
    for idx, w in enumerate(w_list):
        start = start - len(w) - 1
        if start < 0:
            return idx
    return '-'


class MSQALoader(Loader):
    """
    MSMARCO v2.1 数据集loader，
    链接: https://github.com/microsoft/MSMARCO-Question-Answering
    该数据集对应的任务是QA，存在以下文件：
    dev_v2.1.json
    dict{
        answer : {'0':['...',...],...}
        passages : {'0':[{is_selected: 0 or 1, passage_text: '...', url:'...'},...],...}
        query : {'0':'...',...}
        query_id : {'0':'...',...}
        query_type : {LOCATION,NUMERIC,PERSON,DESCRIPTION,ENTITY}
        wellFormedAnswers : {'0': '...'}
    }

    todo: 读取eval
    eval_v2.1_public.json
    dict{
        passages : {'0':[{passage_text: '...', url:'...'},...],...}
        query : {'0':'...',...}
        query_id : {'0':'...',...}
        query_type : {LOCATION,NUMERIC,PERSON,DESCRIPTION,ENTITY}
    }
    """

    def __init__(self):
        super().__init__()
        self.train_filename = ""
        self.dev_file = ""
        self.test_file = ""

    def _load(self, path, **kwargs):
        ds = DataSet()

        if "eval" not in path:
            with open(path, 'r', encoding='utf-8') as fd:
                data = json.load(fd)
                t = tqdm(data['query'])
                t.set_description("load train and dev data")
                for i in t:
                    idx = i
                    query = data['query'][idx]
                    passages = data['passages'][idx]
                    query_id = data['query_id'][idx]
                    query_type = data['query_type'][idx]
                    for passage in passages:
                        passage_text = passage['passage_text']
                        target = passage['is_selected']
                        passage_url = passage['url']
                        ds.append(Instance(
                            raw_words1=query,
                            raw_words2=passage_text,
                            target=target,
                            query_id=query_id,
                            query_type=query_type,
                            passage_url=passage_url
                        ))
        else:
            with open(path, 'r', encoding='utf-8') as fd:
                data = json.load(fd)
                t = tqdm(data['query'])
                t.set_description("load eval data")
                for i in t:
                    idx = i
                    query = data['query'][idx]
                    passages = data['passages'][idx]
                    query_id = data['query_id'][idx]
                    query_type = data['query_type'][idx]
                    for passage in passages:
                        passage_text = passage['passage_text']
                        passage_url = passage['url']
                        ds.append(Instance(
                            raw_words1=query,
                            raw_words2=passage_text,
                            query_id=query_id,
                            query_type=query_type,
                            passage_url=passage_url
                        ))
        return ds

    def download(self):
        raise NotImplementedError

    def load(self, paths=None):
        r"""
        :param str paths: 传入数据所在目录，会在该目录下寻找dev_v2.1.json, eval_v2.1_public.json, train_v2.1.json文件夹
        :return: DataBundle
        """
        if paths:
            paths = os.path.abspath(os.path.expanduser(paths))
        else:
            raise NotImplementedError(
                f"We not support download this data, please download it from https://github.com/microsoft/MSMARCO-Question-Answering")

        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")

        files = {'dev': "dev_v2.1.json",
                 "test": "eval_v2.1_public.json",
                 "train": 'train_v2.1.json'}

        datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if 'test' not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            datasets[name] = self._load(filepath)

        data_bundle = DataBundle(datasets=datasets)

        return data_bundle


class SQuADLoader(Loader):
    """
    SQuAD2.0 数据集dataloader
    链接 https://rajpurkar.github.io/SQuAD-explorer/
    """

    def __init__(self):
        super().__init__()
        self.train_file = ""
        self.dev_file = ""
        self.test_file = ""

    def _load(self, path: str) -> DataSet:
        ds = DataSet()

        with open(path, 'r', encoding='utf-8') as fd:
            file_json = json.load(fd)
            data_json = file_json['data']
            for data in data_json:
                title = data['title']
                for para in data['paragraphs']:
                    context = para['context']
                    for qas in para['qas']:
                        question = qas['question']
                        id = qas['id']
                        for answer in qas['answers']:
                            text = answer['text']
                            answer_start_char = answer['answer_start']

                            ds.append(Instance(id=id,
                                               title=title,
                                               raw_words2=context,
                                               raw_words1=question,
                                               answer=text,
                                               answer_start_char=answer_start_char))

        return ds

    def load(self, paths: Union[str, Dict[str, str]] = None) -> DataBundle:
        r"""
               :param str paths: 传入数据所在目录，会在该目录下寻找dev_v2.1.json, eval_v2.1_public.json, train_v2.1.json文件夹
               :return: DataBundle
               """
        if paths:
            paths = os.path.abspath(os.path.expanduser(paths))
        else:
            raise NotImplementedError(
                f"We not support download this data, please download it from https://github.com/microsoft/MSMARCO-Question-Answering")

        if not os.path.isdir(paths):
            raise NotADirectoryError(f"{paths} is not a valid directory.")

        files = {'dev': "dev-v2.0.json",
                 "train": 'train-v2.0.json'}

        datasets = {}
        for name, filename in files.items():
            filepath = os.path.join(paths, filename)
            if not os.path.isfile(filepath):
                if 'test' not in name:
                    raise FileNotFoundError(f"{name} not found in directory {filepath}.")
            datasets[name] = self._load(filepath)

        data_bundle = DataBundle(datasets=datasets)

        return data_bundle


if __name__ == '__main__':
    a = SQuADLoader().load("/remote-home/zyfei/project/data/squadv2")
    print(a)
