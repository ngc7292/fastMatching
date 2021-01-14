# -*- coding: utf-8 -*-
"""
__title__="matching_esim"
__author__="ngc7293"
__mtime__="2020/11/23"
"""

import random
import numpy as np
import torch
import argparse
from torch.optim import Adamax
from torch.optim.lr_scheduler import StepLR

from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import GradientClipCallback, LRScheduler, EvaluateCallback
from fastNLP.core.losses import CrossEntropyLoss
from fastNLP.embeddings import StaticEmbedding
from fastNLP.embeddings import ElmoEmbedding
from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, MNLIPipe, QNLIPipe, QuoraPipe
from fastNLP import cache_results

from model.esim import ESIM


# define hyper-parameters
class ESIMConfig:
    def __init__(self):
        argument = argparse.ArgumentParser()
        argument.add_argument("--dataset", choices=['snli', 'qnli', 'mnli'], default='snli')
        argument.add_argument("--embedding", choices=['glove', 'word2vec'], default='glove')
        argument.add_argument("--device", type=int, default=0)

        arg = vars(argument.parse_args())

        self.dataset = arg['dataset']
        self.embedding = arg['embedding']

        self.batch_size_per_gpu = 196
        self.n_epochs = 30
        self.lr = 2e-3
        self.seed = 42
        self.save_path = f"../models/{self.dataset}/"  # 模型存储的位置，None表示不存储模型。

        self.train_dataset_name = 'train'
        self.dev_dataset_name = 'dev'
        self.test_dataset_name = 'test'

        self.device = [arg['device']]

        self.to_lower = True  # 忽略大小写
        self.tokenizer = 'spacy'  # 使用spacy进行分词


arg = ESIMConfig()

# set random seed
random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

# n_gpu = torch.cuda.device_count()
device = arg.device
device_count = len(device)
n_gpu = len(device)

if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)

print(f"traing dataset {arg.dataset}, and using {n_gpu} gpus which is {arg.device}")


@cache_results(_cache_fp="../cache/snli", _refresh=True)
def load_snli(args):
    return SNLIPipe(lower=args.to_lower, tokenizer=args.tokenizer).process_from_file()


@cache_results(_cache_fp="../cache/rte", _refresh=True)
def load_rte(args):
    return RTEPipe(lower=args.to_lower, tokenizer=args.tokenizer).process_from_file()


@cache_results(_cache_fp="../cache/qnli", _refresh=True)
def load_qnli(args):
    return QNLIPipe(lower=args.to_lower, tokenizer=args.tokenizer).process_from_file()


@cache_results(_cache_fp="../cache/mnli", _refresh=True)
def load_mnli(args):
    return MNLIPipe(lower=args.to_lower, tokenizer=args.tokenizer).process_from_file()


# load data set
if arg.dataset == 'snli':
    data_bundle = load_snli(arg)
elif arg.dataset == 'qnli':
    data_bundle = load_qnli(arg)
    arg.dev_dataset_name = "dev"

elif arg.dataset == 'mnli':
    data_bundle = load_mnli(arg)
    arg.test_dataset_name = "dev_matched"
    arg.dev_dataset_name = "dev_matched"
else:
    raise RuntimeError(f'NOT support {arg.dataset} dataset yet!')

print(data_bundle)  # print details in data_bundle

# load embedding
if arg.embedding == 'elmo':
    embedding = ElmoEmbedding(data_bundle.vocabs[Const.INPUTS(0)], model_dir_or_name='en-medium',
                              requires_grad=True)
elif arg.embedding == 'word2vec':
    embedding = StaticEmbedding(data_bundle.vocabs[Const.INPUTS(0)], model_dir_or_name='en-word2vec-300',
                                requires_grad=True)
elif arg.embedding == 'glove':
    embedding = StaticEmbedding(data_bundle.vocabs[Const.INPUTS(0)], model_dir_or_name='en-glove-840b-300d',
                                requires_grad=True, normalize=False)
else:
    raise RuntimeError(f'NOT support {arg.embedding} embedding yet!')

# define model
model = ESIM(embedding, num_labels=len(data_bundle.vocabs[Const.TARGET]))

# define optimizer and callback
optimizer = Adamax(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)  # 每10个epoch学习率变为原来的0.5倍

callbacks = [
    GradientClipCallback(clip_value=10),  # 等价于torch.nn.utils.clip_grad_norm_(10)
    LRScheduler(scheduler),
]

if arg.dataset in ['snli']:
    callbacks.append(EvaluateCallback(data=data_bundle.datasets[arg.test_dataset_name]))
    # evaluate test set in every epoch if dataset is snli.

# define trainer
trainer = Trainer(train_data=data_bundle.datasets[arg.train_dataset_name], model=model,
                  optimizer=optimizer,
                  loss=CrossEntropyLoss(),
                  batch_size=device_count * arg.batch_size_per_gpu,
                  n_epochs=arg.n_epochs, print_every=-1,
                  dev_data=data_bundle.datasets[arg.dev_dataset_name],
                  metrics=AccuracyMetric(), metric_key='acc',
                  # device=[i for i in range(torch.cuda.device_count())],
                  device=device,
                  check_code_level=-1,
                  save_path=arg.save_path,
                  callbacks=callbacks)

# train model
trainer.train(load_best_model=True)

# define tester
tester = Tester(
    data=data_bundle.datasets[arg.test_dataset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=device_count * arg.batch_size_per_gpu,
    # device=[i for i in range(torch.cuda.device_count())],
    device=device
)

# test model
tester.test()
