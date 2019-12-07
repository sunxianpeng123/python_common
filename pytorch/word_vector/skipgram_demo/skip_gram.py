# encoding: utf-8

"""
@author: sunxianpeng
@file: skip_gram.py
@time: 2019/11/7 16:25
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from torch.nn.parameter import Parameter

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

def word_tokenize(text):
    """# tokenize函数，把一篇文本转化成一个个单词"""
    return text.split()
class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        ''' text: a list of words, all text from the training dataset
            word_to_idx: the dictionary from word to idx
            idx_to_word: idx to word mapping
            word_freq: the frequency of each word
            word_counts: the word counts
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(t, VOCAB_SIZE - 1) for t in text]
        self.text_encoded = torch.Tensor(self.text_encoded).long()
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        ''' 返回整个数据集（所有单词）的长度
        '''
        return len(self.text_encoded)

    def __getitem__(self, idx):
        ''' 这个function返回以下数据用于训练
            - 中心词
            - 这个单词附近的(positive)单词
            - 随机采样的K个单词作为negative sample
        '''
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx - C, idx)) + list(range(idx + 1, idx + C + 1))
        pos_indices = [i % len(self.text_encoded) for i in pos_indices]
        pos_words = self.text_encoded[pos_indices]
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True)

        return center_word, pos_words, neg_words
class EmbeddingModel(nn.Module):
    '''
    初始化输出和输出embedding
    '''
    def __init__(self,vocab_size,embed_size):
        super(EmbeddingModel,self).__init__()
        self.vocab_size = vocab_size#30000
        self.embed_size = embed_size#100

        initrange = 0.5 / self.embed_size
        # 模型输出nn.Embedding(30000, 100)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 权重初始化的一种方法
        self.out_embed.weight.data.uniform_(-initrange, initrange)

        # 模型输入nn.Embedding(30000, 100)
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=False)
        # 权重初始化的一种方法
        self.in_embed.weight.data.uniform_(-initrange, initrange)


    def forward(self, input_labels,pos_labels,neg_labels):
        '''
        input_labels: 中心词, [batch_size]
        pos_labels: 中心词周围 context window 出现过的单词 [batch_size * (window_size * 2)]
        neg_labelss: 中心词周围没有出现过的单词，从 negative sampling 得到 [batch_size, (window_size * 2 * K)]
        return: loss, [batch_size]
        '''
        # loss function
        # input_labels是输入的标签，tud.DataLoader()返回的。相已经被分成batch了。
        batch_size = input_labels.size(0)
        #B * embed_size,这里估计进行了运算：（128,30000）*（30000,100）= 128(B) * 100 (embed_size)
        input_embedding = self.in_embed(input_labels)  # B * embed_size
        # 同上，增加了维度(2*C)，表示一个batch有B组周围词单词，一组周围词有(2*C)个单词，每个单词有embed_size个维度。
        pos_embedding = self.out_embed(pos_labels)  # B * (2*C) * embed_size
        # 同上，增加了维度(2*C*K)
        neg_embedding = self.out_embed(neg_labels)  # B * (2*C * K) * embed_size
        # bmm ：矩阵{b*n*m}和{b*m*p}相乘得到{b*n*p}
        log_pos = torch.bmm(pos_embedding, input_embedding.unsqueeze(2)).squeeze()  # B * (2*C)
        log_neg = torch.bmm(neg_embedding, -input_embedding.unsqueeze(2)).squeeze()  # B * (2*C*K)

        log_pos = F.logsigmoid(log_pos).sum(1)
        log_neg = F.logsigmoid(log_neg).sum(1)  # batch_size

        loss = log_pos + log_neg

        return -loss


if __name__ == '__main__':
    USE_CUDA = torch.cuda.is_available()
    # 为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
    random.seed(53113)
    np.random.seed(53113)
    torch.manual_seed(53113)
    if USE_CUDA:
        torch.cuda.manual_seed(53113)
    # 设定一些超参数

    K = 100  # number of negative samples 负样本随机采样数量
    C = 3  # nearby words threshold 指定周围三个单词进行预测
    NUM_EPOCHS = 2  # The number of epochs of training
    MAX_VOCAB_SIZE = 30000  # the vocabulary size
    BATCH_SIZE = 128  # the batch size 每轮迭代1个batch的数量
    LEARNING_RATE = 0.2  # the initial learning rate
    EMBEDDING_SIZE = 100#词向量维度

    LOG_FILE = "word-embedding.log"
    with open("text8.train.txt", "r") as fin:
        text = fin.read()
    text = [w for w in word_tokenize(text.lower())]
    vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE - 1))
    vocab["<unk>"] = len(text) - np.sum(list(vocab.values()))
    idx_to_word = [word for word in vocab.keys()]
    word_to_idx = {word: i for i, word in enumerate(idx_to_word)}

    word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3. / 4.)
    word_freqs = word_freqs / np.sum(word_freqs)  # 用来做 negative sampling
    VOCAB_SIZE = len(idx_to_word)
    print("vocabulary size = {}".format(VOCAB_SIZE))
    dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
    dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)














