# FastMatching

## what's this project
这里使用fastNLP复现了几个著名的Text Matching任务的模型，旨在达到与论文中相符的性能。

## about datasets
本项目中整理了部分Text Matching任务的数据集，具体数据集细节整理在下表：

Name| Paper | Description 
:---: | :---: | :---: 
MNLI | - | -
SNLI | - | -
RTE | - | -
QNLI | - | -
Quora | - | -
MSMARCO(Ranking) | - | -
MSMARCO(Question Anwsering) | - | -
SQuADv2.0| [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf) | -
ASNQ | [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection](https://arxiv.org/pdf/1911.04118.pdf) | -

## about models and how to run it 

复现的模型有（按论文发表时间顺序排序）:
- CNTN：[训练代码](matching_cntn.py).
论文链接：[Convolutional Neural Tensor Network Architecture for Community-based Question Answering](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844). 

- ESIM：[训练代码](matching_esim.py).
论文链接：[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf).

- MwAN：[训练代码](matching_mwan.py).
论文链接：[Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/proceedings/2018/0613.pdf).

- BERT： [训练代码](matching_bert.py).
论文链接：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).

## about results
使用fastNLP复现的结果vs论文汇报结果，在前面的表示使用fastNLP复现的结果

下面模型的主要指标为acc

'\-'表示我们仍未复现或者论文原文没有汇报

model name | SNLI | MNLI | RTE | QNLI | Quora
:---: | :---: | :---: | :---: | :---: | :---:
CNTN [代码](model/cntn.py); [论文](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844) | 77.79 vs - | 63.29/63.16(dev) vs - | 57.04(dev) vs - | 62.38(dev) vs - | - |
ESIM[代码](model/bert.py); [论文](https://arxiv.org/pdf/1609.06038.pdf) | 88.13(glove) vs 88.0(glove)/88.7(elmo) | 77.78/76.49 vs 72.4/72.1* | 59.21(dev) vs - | 76.97(dev) vs - | - |
MwAN [代码](model/mwan.py); [论文](https://www.ijcai.org/proceedings/2018/0613.pdf) | 87.9 vs 88.3 | 77.3/76.7(dev) vs 78.5/77.7 | - | 74.6(dev) vs - | 85.6 vs 89.12 |
BERT (BASE version)[代码](model/bert.py); [论文](https://arxiv.org/pdf/1810.04805.pdf) | 90.6 vs - | - vs 84.6/83.4| 67.87(dev) vs 66.4 | 90.97(dev) vs 90.5 | - |

*ESIM模型由MNLI官方复现的结果为72.4/72.1，ESIM原论文当中没有汇报MNLI数据集的结果。

### TODO

- [ ] 迁移数据loader以及model 

- [ ] DC-BERT

- [ ] ColBERT

- [ ] DeFormer

- [ ] 补充readme