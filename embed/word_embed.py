# -*- coding: utf-8 -*-

"""
Generate word embeddings
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

from model import WordEmbedding

WORD_EMBEDDING_DIM = 48
WINDOW_SIZE = 5

with open('config.json', 'r') as fin:
    args = json.loads(fin.read())

with open(args['dataset_path'], 'rb') as fin:
    dataset = pickle.load(fin)
with open(args['lookup_path'], 'rb') as fin:
    lookup = pickle.load(fin)

sentences = dataset['sens_train'] + dataset['sens_val'] + dataset['sens_test']  # (num_sentences, VAR_len)
vocab_size = len(lookup['word2idx'])


# Build skip-grams
def build_skip_grams(corpus, window_size):
    """Build skip grams with given window size

    Args:
        corpus (list[list[str]]): corpus including word tokens
        window_size (int): window size

    Returns:
        (list[tuple[int, int]]): skip grams
    """
    skip_grams = []

    for tokens in corpus:
        for i, center in enumerate(tokens):
            left = max(0, i - window_size)
            right = min(i + window_size, len(tokens))
            for j in range(left, right):
                if j != i:
                    skip_grams.append((center, tokens[j]))

    return skip_grams


skip_grams = build_skip_grams(sentences, WINDOW_SIZE)


def gen_batch(data, batch_size):
    """Generator to give random batches from skip-grams

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
            center_batch = batch[:, 0]
            context_batch = batch[:, 1]

            yield center_batch, context_batch
            break

        else:
            batch = data[indices[idx: idx + batch_size]]
            center_batch = batch[:, 0]
            context_batch = batch[:, 1]

            yield center_batch, context_batch
            idx += batch_size


if __name__ == '__main__':
    if not os.path.exists(args['embed_dir']):
        os.makedirs(args['embed_dir'])

    # Hyperparameters
    lr = 0.001
    batch_size = 64
    num_epochs = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = WordEmbedding(vocab_size, WORD_EMBEDDING_DIM).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print('Start training word embeddings')
    model.train()

    for epoch in range(1, num_epochs + 1):
        for i, (centers, contexts) in enumerate(gen_batch(skip_grams, batch_size)):
            centers = torch.from_numpy(centers).long().to(device)
            contexts = torch.from_numpy(contexts).long().to(device)

            out = model(centers)
            loss = criterion(out, contexts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 1000 == 0:
                print('Epoch: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))

        np.save(os.path.join(args['embed_dir'], 'word_{:0>2d}.npy'.format(epoch)), model.embedding.weight.detach().cpu().numpy())
        if os.path.exists(os.path.join(args['embed_dir'], 'word_{:0>2d}.npy'.format(epoch - 5))):
            os.remove(os.path.join(args['embed_dir'], 'word_{:0>2d}.npy'.format(epoch - 5)))

    np.save(args['embed_word_path'], model.embedding.weight.detach().cpu().numpy())
    # Remove temp files
    for ftemp in glob.glob(os.path.join(args['embed_dir'], 'word_*.npy')):
        if os.path.exists(ftemp):
            os.remove(ftemp)

    print('Finished training word embeddings')
