# -*- coding: utf-8 -*-
"""
__title__="matching_bert"
__author__="ngc7293"
__mtime__="2020/11/24"
"""
import random
import numpy as np
import torch
import argparse

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import WarmupCallback, EvaluateCallback
from fastNLP.core.optimizer import AdamW
from fastNLP.embeddings import BertEmbedding
from fastNLP.models.bert import BertForSentenceMatching
from fastNLP import LossInForward

from load_data.load_data import load_mnli_bert, load_qnli_bert, load_rte_bert, load_snli_bert


# define hyper-parameters
class BERTConfig:

    def __init__(self):
        argument = argparse.ArgumentParser()
        argument.add_argument("--dataset", choices=['snli', 'qnli', 'mnli'], default='snli')
        argument.add_argument("--device", type=list, default=0)

        argument = argument.parse_args()

        self.dataset = argument.dataset
        self.device = argument.device

        self.batch_size_per_gpu = 6
        self.n_epochs = 6
        self.lr = 2e-5
        self.warm_up_rate = 0.1
        self.seed = 42
        self.save_path = None  # 模型存储的位置，None表示不存储模型。

        self.train_dataset_name = 'train'
        self.dev_dataset_name = 'dev'
        self.test_dataset_name = 'test'

        self.lower = True  # 忽略大小写
        self.tokenizer = 'spacy'  # 使用spacy进行分词

        self.bert_model_dir_or_name = 'en-base-uncased'


arg = BERTConfig()

# set random seed
random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

# n_gpu = torch.cuda.device_count()
n_gpu = len(arg.device)

if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)

print(f"traing dataset { arg.dataset }, and using { n_gpu } gpus which is {arg.device}")

# load data set
cache_name = 'cache/{}_{}_{}'.format(arg.dataset, arg.lower, arg.tokenizer)
refresh_data = False
if arg.dataset == 'snli':
    data_bundle = load_snli_bert(lower=arg.lower, tokenizer=arg.tokenizer, _cache_fp=cache_name, refresh_=refresh_data)
elif arg.dataset == 'rte':
    data_bundle = load_rte_bert(lower=arg.lower, tokenizer=arg.tokenizer, _cache_fp=cache_name, refresh_=refresh_data)
elif arg.dataset == 'qnli':
    data_bundle = load_qnli_bert(lower=arg.lower, tokenizer=arg.tokenizer, _cache_fp=cache_name, refresh_=refresh_data)
elif arg.dataset == 'mnli':
    data_bundle = load_mnli_bert(lower=arg.lower, tokenizer=arg.tokenizer, _cache_fp=cache_name, refresh_=refresh_data)
else:
    raise RuntimeError(f'NOT support {arg.dataset} dataset yet!')

print(data_bundle)  # print details in data_bundle

# load embedding
embed = BertEmbedding(data_bundle.vocabs[Const.INPUT], model_dir_or_name=arg.bert_model_dir_or_name)

# define model
model = BertForSentenceMatching(embed, num_labels=len(data_bundle.vocabs[Const.TARGET]))

# define optimizer and callback
optimizer = AdamW(lr=arg.lr, params=model.parameters())
callbacks = [WarmupCallback(warmup=arg.warm_up_rate, schedule='linear'), ]

if arg.dataset in ['snli']:
    callbacks.append(EvaluateCallback(data=data_bundle.datasets[arg.test_dataset_name]))
    # evaluate test set in every epoch if dataset is snli.

# define trainer
trainer = Trainer(train_data=data_bundle.get_dataset(arg.train_dataset_name), model=model,
                  optimizer=optimizer,
                  batch_size=n_gpu * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_bundle.get_dataset(arg.dev_dataset_name),
                  metrics=AccuracyMetric(), metric_key='acc',
                  # device=[i for i in range(torch.cuda.device_count())],
                  loss=LossInForward(),
                  device=arg.device,
                  check_code_level=-1,
                  save_path=arg.save_path,
                  callbacks=callbacks)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_bundle.get_dataset(arg.test_dataset_name),
    model=model,
    metrics=AccuracyMetric(),
    batch_size=n_gpu * arg.batch_size_per_gpu,
    device=arg.device,
)

# test model
tester.test()
