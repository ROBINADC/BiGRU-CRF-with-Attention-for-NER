# -*- coding: utf-8 -*-

"""
Models for generating different feature embeddings
Created on 2020/5/30 
"""

__author__ = "Yihang Wu"

import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, vocab_size, bias=False)

        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        """Doing skip-gram word2vec

        Args:
            x (torch.LongTensor): input words with shape (B,)

        Returns:
            (torch.Tensor): output predicted words with shape (B,)
        """
        x = self.embedding(x)  # (B, embedding_size)
        x = self.fc(x)  # (B, vocab_size)
        return F.log_softmax(x, dim=1)  # (B,)


class CharEmbedding(nn.Module):
    """Character embeddings
    Last hidden states of forward and backward lstm will be extracted and concatenated as character embedding of a word
    """
    def __init__(self, char_size, embedding_size, h_size):
        super().__init__()

        self.embedding = nn.Embedding(char_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, h_size, batch_first=True, bidirectional=True)

        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        # x (B, VAR_word_len)
        x = self.embedding(x)  # (B, VAR_len, embedding_size)
        _, (h, _) = self.lstm(x)  # (2, B, h_size)
        out = torch.cat(torch.chunk(h, h.size(0), dim=0), dim=2).squeeze(dim=0)  # (B, h_size * 2)
        return out


class PoSEmbedding(nn.Module):
    """Part-of-Speech embeddings
    Trained with part-of-speech tag, the embedding layer will be the PoS embeddings
    """
    def __init__(self, vocab_size, embedding_size, num_entities):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.fc = nn.Linear(embedding_size, num_entities)

        nn.init.xavier_normal_(self.embedding.weight)

    def forward(self, x):
        # x (B,)
        x = self.embedding(x)  # (B, embedding_size)
        x = self.fc(x)  # (B, num_entities)
        return x
