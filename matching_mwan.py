# -*- coding: utf-8 -*-
import random
import argparse
import fitlog
import numpy as np
import torch
from torch.optim import Adadelta
from torch.optim.lr_scheduler import StepLR
from fastNLP import CrossEntropyLoss
from fastNLP.core import Trainer, Tester, AccuracyMetric, Const
from fastNLP.core.callback import LRScheduler, EvaluateCallback
from fastNLP.embeddings import StaticEmbedding
from fastNLP.io.pipe.matching import SNLIPipe, RTEPipe, MNLIPipe, QNLIPipe, QuoraPipe
from model.mwan import MwanModel

fitlog.debug()


class MWANConfig:
    def __init__(self):
        argument = argparse.ArgumentParser()
        argument.add_argument('--dataset', choices=['snli', 'rte', 'qnli', 'mnli'], default='snli')
        argument.add_argument('--batch-size', type=int, default=128)
        argument.add_argument('--n-epochs', type=int, default=50)
        argument.add_argument('--lr', type=float, default=1)
        argument.add_argument('--seed', type=int, default=42)
        argument.add_argument('--hidden-size', type=int, default=150)
        argument.add_argument('--dropout', type=float, default=0.3)
        arg = argument.parse_args()

        self.dataset = arg.dataset

        self.batch_size = arg.batch_size
        self.n_epochs = arg.n_epochs
        self.lr = arg.lr
        self.seed = arg.seed
        self.hidden_size = arg.hidden_size
        self.dropout = arg.dropout

        self.train_dataset_name = 'train'
        self.dev_dataset_name = 'dev'
        self.test_dataset_name = 'test'


arg = MWANConfig()

random.seed(arg.seed)
np.random.seed(arg.seed)
torch.manual_seed(arg.seed)

n_gpu = torch.cuda.device_count()
if n_gpu > 0:
    torch.cuda.manual_seed_all(arg.seed)
print(n_gpu)

for k in arg.__dict__:
    print(k, arg.__dict__[k], type(arg.__dict__[k]))

# load data set
if arg.dataset == 'snli':
    data_bundle = SNLIPipe(lower=True, tokenizer='spacy').process_from_file()
elif arg.dataset == 'rte':
    data_bundle = RTEPipe(lower=True, tokenizer='spacy').process_from_file()
elif arg.dataset == 'qnli':
    data_bundle = QNLIPipe(lower=True, tokenizer='spacy').process_from_file()
elif arg.dataset == 'mnli':
    data_bundle = MNLIPipe(lower=True, tokenizer='spacy').process_from_file()
elif arg.dataset == 'quora':
    data_bundle = QuoraPipe(lower=True, tokenizer='spacy').process_from_file()
else:
    raise RuntimeError(f'NOT support {arg.dataset} dataset yet!')

print(data_bundle)
print(len(data_bundle.vocabs[Const.INPUTS(0)]))

model = MwanModel(
    num_class=len(data_bundle.vocabs[Const.TARGET]),
    EmbLayer=StaticEmbedding(data_bundle.vocabs[Const.INPUTS(0)], requires_grad=False, normalize=False),
    ElmoLayer=None,
    args_of_imm={
        "input_size": 30,
        "hidden_size": arg.hidden_size,
        "dropout": arg.dropout,
        "use_allennlp": False, },
)

optimizer = Adadelta(lr=arg.lr, params=model.parameters())
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

callbacks = [LRScheduler(scheduler)]

if arg.dataset in ['snli']:
    callbacks.append(EvaluateCallback(data=data_bundle.datasets[arg.test_dataset_name]))
elif arg.dataset == 'mnli':
    callbacks.append(EvaluateCallback(data={'dev_matched': data_bundle.datasets['dev_matched'],
                                            'dev_mismatched': data_bundle.datasets['dev_mismatched']}, ))

trainer = Trainer(
    train_data=data_bundle.datasets[arg.train_dataset_name],
    model=model,
    optimizer=optimizer,
    num_workers=0,
    batch_size=arg.batch_size,
    n_epochs=arg.n_epochs,
    print_every=-1,
    dev_data=data_bundle.datasets[arg.dev_dataset_name],
    metrics=AccuracyMetric(pred="pred", target="target"),
    metric_key='acc',
    device=[i for i in range(torch.cuda.device_count())],
    check_code_level=-1,
    callbacks=callbacks,
    loss=CrossEntropyLoss(pred="pred", target="target")
)
trainer.train(load_best_model=True)

tester = Tester(
    data=data_bundle.datasets[arg.test_dataset_name],
    model=model,
    metrics=AccuracyMetric(),
    batch_size=arg.batch_size,
    device=[i for i in range(torch.cuda.device_count())],
)
tester.test()
