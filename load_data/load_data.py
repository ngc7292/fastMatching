# -*- coding: utf-8 -*-
from fastNLP.io.pipe import SNLIPipe, QNLIPipe, MNLIPipe, RTEPipe
from fastNLP import cache_results

from fastNLP.io.pipe import SNLIBertPipe, QNLIBertPipe, RTEBertPipe, MNLIBertPipe


@cache_results('cache/snli', _refresh=False)
def load_snli(lower=False, tokenizer='raw'):
    bundle = SNLIPipe(lower=lower, tokenizer=tokenizer).process_from_file()
    return bundle


@cache_results('cache/qnli')
def load_qnli(lower=False, tokenizer='raw'):
    return QNLIPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/mnli')
def load_mnli(lower=False, tokenizer='raw'):
    return MNLIPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/rte')
def load_rte(lower=False, tokenizer='raw'):
    return RTEPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/snli_bert')
def load_snli_bert(lower=False, tokenizer='raw'):
    return SNLIBertPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/qnli_bert')
def load_qnli_bert(lower=False, tokenizer='raw'):
    return QNLIBertPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/mnli_bert')
def load_mnli_bert(lower=False, tokenizer='raw'):
    return MNLIBertPipe(lower=lower, tokenizer=tokenizer).process_from_file()


@cache_results('cache/rte_bert')
def load_rte_bert(lower=False, tokenizer='raw'):
    return RTEBertPipe(lower=lower, tokenizer=tokenizer).process_from_file()
