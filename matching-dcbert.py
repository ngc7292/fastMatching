# -*- coding: utf-8 -*-
"""
__title__="matching-dcbert"
__author__="ngc7293"
__mtime__="2021/1/14"
"""
import argparse
import torch
import fitlog

from torch import optim
from load_data.load_data import load_mnli, load_qnli, load_snli, load_rte
from model.dcbert import BERT_Matching
from fastNLP import AccuracyMetric, FitlogCallback, WarmupCallback, Trainer, Tester, LossInForward

fitlog.set_log_dir("logs")

class DCBERTConfig:
    def __init__(self):
        args = argparse.ArgumentParser()
        args.add_argument("--dataset", default='snli', choices=['snli', 'qnli', 'mnli', 'msqa'])
        args.add_argument("--train_dataset_name", default="train", type=str)
        args.add_argument("--dev_dataset_name", default="dev", type=str)
        args.add_argument("--test_dataset_name", default="test", type=str)

        args.add_argument("--model_dir_or_name", default="en-base-cased")
        args.add_argument('--pool_method', default='first', choices=['last', 'first', 'avg', 'max'])
        args.add_argument("--decoder_mode", default="linnear", choices=['linnear', 'mutli'])
        args.add_argument("--bert_dropout", default=0.5, type=float)
        args.add_argument("--decoder_dropout", default=0.1, type=float)

        args.add_argument("--transformer_encoder_layer", default=6, type=int)
        args.add_argument("--transformer_dim", default=128, type=int)
        args.add_argument("--transformer_num_head", default=8, type=int)
        args.add_argument("--transformer_dropout", default=0.5, type=float)

        args.add_argument("--optimizer", default="adam", choices=['adam', 'sgd'])
        args.add_argument("--lr", default=1e-4, type=float)
        args.add_argument("--ptm_lr_rate", default=0.1, type=float)
        args.add_argument('--weight_decay', default=1e-2, type=float)
        args.add_argument('--momentum', default=0.9, type=float)
        args.add_argument('--batch_size', default=32, type=int)
        args.add_argument('--epoch', default=10, type=int)
        args.add_argument('--fix_ptm_epoch', default=-1, type=int)
        args.add_argument('--warmup_step', default=0.1, type=float)
        args.add_argument('--warmup_schedule', default='linear')

        args.add_argument('--device', default='0')
        args.add_argument('--lower', type=int, default=0)
        args.add_argument('--tokenizer', default='spacy', choices=['raw', 'spacy'])

        args.add_argument('--debug', action='store_true')

        self.args = args.parse_args()

        if self.args.dataset == "mnli":
            self.args.dev_dataset_name = "dev_matched"
            self.args.test_dataset_name = "test_matched"

    def get_args(self):
        return self.args


args = DCBERTConfig().get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.device = device
args.n_gpu = torch.cuda.device_count()

print("====================================")
print(f"device: {device}, n_gpu: {args.n_gpu}")
print("====================================")

print("loading datasets ...")

cache_name = 'cache/{}_{}_{}'.format(args.dataset, args.lower, args.tokenizer)
refresh_data = False
if args.dataset == 'snli':
    bundle = load_snli(args.lower, args.tokenizer, _cache_fp=cache_name, _refresh=refresh_data)
elif args.dataset == 'qnli':
    bundle = load_qnli(args.lower, args.tokenizer, _cache_fp=cache_name, _refresh=refresh_data)
elif args.dataset == 'mnli':
    bundle = load_mnli(args.lower, args.tokenizer, _cache_fp=cache_name, _refresh=refresh_data)
    test_dataset_key = "dev"
elif args.dataset == 'rte':
    bundle = load_rte(args.lower, args.tokenizer, _cache_fp=cache_name, _refresh=refresh_data)
else:
    raise NotImplementedError

if args.debug:
    bundle.datasets[args.train_dataset_name] = bundle.datasets[args.train_dataset_name][:1000]
    bundle.datasets[args.dev_dataset_name] = bundle.datasets[args.train_dataset_name][:1000]
    bundle.datasets[args.test_dataset_name] = bundle.datasets[args.train_dataset_name][:1000]

print(bundle)

model = BERT_Matching(dataBundle=bundle, args=args)

print(bundle.vocabs.keys())
print(bundle.vocabs['target'].word2idx)
for k, v in bundle.datasets.items():
    if v.has_field('target'):
        v.set_input('words1', 'words2', 'seq_len1', 'seq_len2', 'target')
        v.set_target('target')

    v.set_pad_val('words1', bundle.get_vocab('words1').padding_idx)
    v.set_pad_val('words2', bundle.vocabs['words1'].padding_idx)

model_param = list(model.parameters())
ptm_param = list(model.ptm_encoder.parameters())
ptm_param_id = list(map(id, ptm_param))

non_ptm_param = list(filter(lambda x: id(x) not in ptm_param_id, model_param))

param_ = [{'params': ptm_param, 'lr': args.lr * args.ptm_lr_rate}, {'params': non_ptm_param, 'lr': args.lr}]

if args.optimizer == 'adam':
    optimizer = optim.AdamW(param_, lr=args.lr, weight_decay=args.weight_decay)
elif args.optimizer == 'sgd':
    optimizer = optim.SGD(param_, lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)
else:
    raise NotImplementedError

metrics = AccuracyMetric(pred='pred', target='target')

if bundle.datasets[args.test_dataset_name].has_field('target'):
    print('{} 有test集'.format(args.dataset))
    fitlog.add_hyper(1, 'has_test')
    fitlog_callback = FitlogCallback(data=bundle.datasets[args.test_dataset_name], verbose=1)
else:
    print('{} 没有test集，所以这里的test是dev集'.format(args.dataset))
    fitlog.add_hyper(0, 'has_test')
    fitlog_callback = FitlogCallback(verbose=1, data=bundle.get_dataset(args.dev_dataset_name)[:1000])

callbacks = [fitlog_callback, ]

if args.warmup_step:
    callbacks.append(WarmupCallback(warmup=args.warmup_step,schedule=args.warmup_schedule))


trainer = Trainer(bundle.datasets[args.train_dataset_name],
                  model,
                  optimizer=optimizer,
                  loss=LossInForward(),
                  batch_size=args.batch_size,
                  n_epochs=args.epoch,
                  dev_data=bundle.datasets[args.dev_dataset_name],
                  metrics=metrics,
                  test_use_tqdm=True,
                  callbacks=callbacks,
                  device=device)

train_return = trainer.train()
print(train_return)

print("model testing ...")
tester = Tester(bundle.datasets[args.test_dataset_name],
                model,
                metrics=metrics,
                use_tqdm=True)

test_return = tester.test()
print(test_return)
