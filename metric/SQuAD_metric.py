# -*- coding: utf-8 -*-
"""
__title__="QAmetric"
__author__="ngc7293"
__mtime__="2021/1/19"
"""
import re
import string
import collections

from fastNLP import MetricBase


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_raw_scores(dataset, preds):
    exact_scores = {}
    f1_scores = {}
    for article in dataset:
        for p in article['paragraphs']:
            for qa in p['qas']:
                qid = qa['id']
                gold_answers = [a['text'] for a in qa['answers']
                                if normalize_answer(a['text'])]
                if not gold_answers:
                    # For unanswerable questions, only correct answer is empty string
                    gold_answers = ['']
                if qid not in preds:
                    print('Missing prediction for %s' % qid)
                    continue
                a_pred = preds[qid]
                # Take max over all gold answers
                exact_scores[qid] = max(compute_exact(a, a_pred) for a in gold_answers)
                f1_scores[qid] = max(compute_f1(a, a_pred) for a in gold_answers)
    return exact_scores, f1_scores


class SQuADv2Metric(MetricBase):
    """
    SQuADv2.0任务的评价Metric
    """

    def __init__(self, pred=None, target=None):
        super().__init__()

        self._init_param_map(pred=pred, target=target)

        self.em = 0
        self.f1 = 0
        self.total = 0

    def evaluate(self, pred, target):
        for p, t in zip(pred, target):
            self.em += compute_exact(a_gold=p, a_pred=t)
            self.f1 += compute_f1(a_gold=p, a_pred=t)
            self.total += 1

    def get_metric(self, reset=True):
        em_res = self.em / self.total
        f1_res = self.f1 / self.total
        if reset:
            self.em = 0
            self.f1 = 0
            self.total = 0
        return {
            'em': em_res,
            'f1': f1_res
        }


if __name__ == '__main__':
    f1 = compute_f1("in the late 1990s", "on the late 1990s")
    em = compute_exact("in the late 1990s", "in the late 1990s")

    print(f1)
    print(em)
