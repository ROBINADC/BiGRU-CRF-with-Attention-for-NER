# -*- coding: utf-8 -*-

"""
Generate Part-of-Speech embeddings
Created on 2020/5/30 
"""

__author__ = "Yihang Wu"

import os
import json
import glob
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import nltk

nltk.download('averaged_perceptron_tagger')

from model import PoSEmbedding

POS_EMBEDDING_DIM = 32

with open('config.json', 'r') as fin:
    args = json.loads(fin.read())

with open(args['dataset_path'], 'rb') as fin:
    dataset = pickle.load(fin)
with open(args['lookup_path'], 'rb') as fin:
    lookup = pickle.load(fin)

word2idx = lookup['word2idx']

vocab_size = len(lookup['word2idx'].keys())

idx2word_lookup = {i: w for w, i in word2idx.items()}
sentences = dataset['sens_train'] + dataset['sens_val'] + dataset['sens_test']
sentences = [[idx2word_lookup[idx] for idx in sens] for sens in sentences]

data = [nltk.pos_tag(sen) for sen in sentences]
tags = set(pair[1] for sen in data for pair in sen)
num_tags = len(tags)

tag2idx = {tag: idx for idx, tag in enumerate(tags)}

data = [(word2idx[word], tag2idx[tag]) for seqs in data for word, tag in seqs]


def gen_batch(data, batch_size):
    """Generator to give random batches from word-entity set

    Args:
        data (list[tuple[int, int]]): data
        batch_size (int): batch size

    Returns:
        (generator)
    """
    data = np.array(data)
    data_len = len(data)
    indices = np.arange(data_len)
    np.random.shuffle(indices)

    idx = 0

    while True:

        if idx + batch_size >= data_len:
            batch = data[indices[idx:]]
            x = batch[:, 0]
            y = batch[:, 1]

            yield x, y
            break

        else:
            batch = data[indices[idx: idx + batch_size]]
            x = batch[:, 0]
            y = batch[:, 1]

            yield x, y
            idx += batch_size


if __name__ == '__main__':
    # Hyperparameters
    lr = 0.001
    batch_size = 100
    num_epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = PoSEmbedding(vocab_size, POS_EMBEDDING_DIM, num_tags).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('Start training part-of-speech embeddings')
    model.train()

    for epoch in range(1, num_epochs + 1):
        for i, (x, y) in enumerate(gen_batch(data, batch_size)):
            x = torch.from_numpy(x).long().to(device)  # (B,)
            y = torch.from_numpy(y).long().to(device)  # (B,)

            out = model(x)  # (B, num_entities=5)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))

        np.save(os.path.join(args['embed_dir'], 'pos_{:0>2d}.npy'.format(epoch)),
                model.embedding.weight.detach().cpu().numpy())
        if os.path.exists(os.path.join(args['embed_dir'], 'pos_{:0>2d}.npy'.format(epoch - 5))):
            os.remove(os.path.join(args['embed_dir'], 'pos_{:0>2d}.npy'.format(epoch - 5)))

    np.save(args['embed_pos_path'], model.embedding.weight.detach().cpu().numpy())
    # Remove temp files
    for ftemp in glob.glob(os.path.join(args['embed_dir'], 'pos_*.npy')):
        if os.path.exists(ftemp):
            os.remove(ftemp)

    print('Finished training part-of-speech embeddings')
