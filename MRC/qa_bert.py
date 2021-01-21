# -*- coding: utf-8 -*-
"""
__title__="qa_bert"
__author__="ngc7293"
__mtime__="2021/1/20"
"""
import sys
sys.path.append("/remote-home/zyfei/project/fastMatching_pro")
# print(sys.path)
import torch
import argparse
import fitlog

from torch import optim
from fastNLP import FitlogCallback, WarmupCallback, Trainer, Tester, LossInForward

from load_data.load_data import load_squad
from model.bert_for_qa import BertForQuestionAnswering
from metric.SQuAD_metric import SQuADv2Metric

fitlog.set_log_dir("/remote-home/zyfei/project/fastMatching_pro/logs")


class DCBERTConfig:
    def __init__(self):
        arg = argparse.ArgumentParser()
        arg.add_argument("--dataset", default='squad', choices=['squad'])
        arg.add_argument("--train_dataset_name", default="train", type=str)
        arg.add_argument("--dev_dataset_name", default="dev", type=str)
        arg.add_argument("--test_dataset_name", default="dev", type=str)

        arg.add_argument("--model_dir_or_name", default="en-base-cased")
        arg.add_argument('--pool_method', default='first', choices=['last', 'first', 'avg', 'max'])
        arg.add_argument('--layers', default='-1', type=str)
        arg.add_argument("--bert_dropout", default=0.5, type=float)

        arg.add_argument("--optimizer", default="adam", choices=['adam', 'sgd'])
        arg.add_argument("--lr", default=1e-4, type=float)
        arg.add_argument("--ptm_lr_rate", default=0.1, type=float)
        arg.add_argument('--weight_decay', default=1e-2, type=float)
        arg.add_argument('--momentum', default=0.9, type=float)
        arg.add_argument('--batch_size', default=32, type=int)
        arg.add_argument('--epoch', default=10, type=int)
        arg.add_argument('--fix_ptm_epoch', default=-1, type=int)
        arg.add_argument('--warmup_step', default=0.1, type=float)
        arg.add_argument('--warmup_schedule', default='linear')

        arg.add_argument('--device', default='0')
        arg.add_argument('--lower', type=int, default=0)
        arg.add_argument('--tokenizer', default='spacy', choices=['raw', 'spacy'])

        arg.add_argument('--debug', action='store_true')

        self.args = arg.parse_args()

    def get_args(self):
        return self.args


args = DCBERTConfig().get_args()

if torch.cuda.is_available() and not args.debug:
    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()
else:
    device = torch.device("cpu")
    args.n_gpu = 0

args.device = device

print("====================================")
print(f"device: {device}, n_gpu: {args.n_gpu}")
print("====================================")

print("loading datasets ...")

refresh_data = False
if args.dataset == "squad":
    bundle = load_squad(lower=args.lower, tokenizer=args.tokenizer, _refresh=refresh_data)
else:
    raise NotImplementedError

if args.debug:
    bundle.datasets[args.train_dataset_name] = bundle.datasets[args.train_dataset_name][:300]
    bundle.datasets[args.dev_dataset_name] = bundle.datasets[args.train_dataset_name][:100]
    bundle.datasets[args.test_dataset_name] = bundle.datasets[args.train_dataset_name][:100]

print(bundle)

for k, v in bundle.datasets.items():
    v.set_input('words', 'seq_len1', 'answer_start', 'answer_end')
    v.set_target('answer')

model = BertForQuestionAnswering(bundle, args)

model_param = list(model.parameters())
ptm_param = list(model.bert.parameters())
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

metrics = SQuADv2Metric()

print('{} 没有test集，所以这里的test是dev集'.format(args.dataset))
fitlog.add_hyper(0, 'has_test')
fitlog_callback = FitlogCallback(verbose=1, data=bundle.get_dataset(args.dev_dataset_name)[:1000])

callbacks = [fitlog_callback, ]

if args.warmup_step:
    callbacks.append(WarmupCallback(warmup=args.warmup_step, schedule=args.warmup_schedule))

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
