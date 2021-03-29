# FastMatching

## what's this project
这里使用fastNLP复现了几个著名的Text Matching任务的模型，旨在达到与论文中相符的性能。

## about datasets
本项目中整理了部分Text Matching任务的数据集，具体数据集细节整理在下表：

Name| Paper | Description 
:---: | :---: | :---: 
MNLI | [A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/pdf/1704.05426.pdf) | 一个文本蕴含的任务，在给定前提（Premise）下，需要判断假设（Hypothesis）是否成立，其中因为MNLI主打卖点是集合了许多不同领域风格的文本，因此又分为matched和mismatched两个版本的MNLI数据集，前者指训练集和测试集的数据来源一致，而后者指来源不一致。该任务属于句子对的文本三分类问题。 
SNLI | [A large annotated corpus for learning natural language inference](https://arxiv.org/pdf/1508.05326.pdf) | -
RTE | - | -
QNLI | [SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/pdf/1606.05250.pdf) | -
Quora | [Natural Language Understanding with the Quora Question Pairs Dataset](https://arxiv.org/pdf/1907.01041.pdf) | -
MSMARCO(Ranking & ReRanking) | [MS MARCO: A HUMAN GENERATED MACHINE READING COMPREHENSION DATASET](https://arxiv.org/pdf/1611.09268.pdf) | -
SQuADv2.0| [Know What You Don't Know: Unanswerable Questions for SQuAD](https://arxiv.org/pdf/1806.03822.pdf) | -
ASNQ | [TANDA: Transfer and Adapt Pre-Trained Transformer Models for Answer Sentence Selection](https://arxiv.org/pdf/1911.04118.pdf) | -

## about models and how to run it 

复现的模型有（按论文发表时间顺序排序）:
- CNTN：[训练代码](IR/matching_cntn.py).
论文链接：[Convolutional Neural Tensor Network Architecture for Community-based Question Answering](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844). 

- ESIM：[训练代码](IR/matching_esim.py).
论文链接：[Enhanced LSTM for Natural Language Inference](https://arxiv.org/pdf/1609.06038.pdf).

- MwAN：[训练代码](IR/matching_mwan.py).
论文链接：[Multiway Attention Networks for Modeling Sentence Pairs](https://www.ijcai.org/proceedings/2018/0613.pdf).

- BERT： [训练代码](IR/matching_bert.py).
论文链接：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf).

- DC-BERT []().
 论文链接：[DC-BERT: DECOUPLING QUESTION AND DOCUMENT FOR EFFICIENT CONTEXTUAL ENCODING](https://arxiv.org/pdf/2002.12591.pdf).
 
 -ColBERT []().
 论文链接：[ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT](https://arxiv.org/abs/2004.12832)

## about results
使用fastNLP复现的结果vs论文汇报结果，在前面的表示使用fastNLP复现的结果

下面模型的主要指标为acc

'\-'表示我们仍未复现或者论文原文没有汇报

model name | SNLI | MNLI | RTE | QNLI | Quora
:---: | :---: | :---: | :---: | :---: | :---:
CNTN  [论文](https://www.aaai.org/ocs/index.php/IJCAI/IJCAI15/paper/view/11401/10844) | 77.79 vs - | 63.29/63.16(dev) vs - | 57.04(dev) vs - | 62.38(dev) vs - | - |
ESIM  [论文](https://arxiv.org/pdf/1609.06038.pdf) | 88.13(glove) vs 88.0(glove)/88.7(elmo) | 77.78/76.49 vs 72.4/72.1* | 59.21(dev) vs - | 76.97(dev) vs - | - |
MwAN  [论文](https://www.ijcai.org/proceedings/2018/0613.pdf) | 87.9 vs 88.3 | 77.3/76.7(dev) vs 78.5/77.7 | - | 74.6(dev) vs - | 85.6 vs 89.12 |
BERT (BASE version) [论文](https://arxiv.org/pdf/1810.04805.pdf) | 90.6 vs - | - vs 84.6/83.4| 67.87(dev) vs 66.4 | 90.97(dev) vs 90.5 | - |
DC-BERT [论文](https://arxiv.org/pdf/2002.12591.pdf)| 70.79 vs - |69.2 vs - | 81.4 vs - | - |
BERT CLS CAT[论文]（） | 70.8 vs -| 69.6 vs - |81.4 vs - |

model name | SQuADv2.0 EM | SQuADv2.0 F1
:---: | :---: | :---:
BERT (BASE version) [论文](https://arxiv.org/pdf/1810.04805.pdf) | - vs 73.302 | - vs 76.284
ALBERT | 88.107 | 90.902

*ESIM模型由MNLI官方复现的结果为72.4/72.1，ESIM原论文当中没有汇报MNLI数据集的结果。



### TODO

- [ ] MSMARCO 数据dataloader

- [ ] SQuAD2.0 dataloader

- [ ] ASNQ dataloader

- [ ] MRR metric

- [ ] Retriever P@10 metric

- [x] DC-BERT

- [ ] ColBERT

- [ ] DeFormer

- [x] 补充readme
