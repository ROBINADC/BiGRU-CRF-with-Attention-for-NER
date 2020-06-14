# -*- coding: utf-8 -*-

"""
Generate character embeddings
Created on 2020/5/30 
"""

__author__ = "Yihang Wu"

import itertools
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from model import CharEmbedding

WORD_EMBEDDING_DIM = 48
CHAR_EMBEDDING_DIM = WORD_EMBEDDING_DIM

with open('config.json', 'r') as fin:
    args = json.loads(fin.read())

with open(args['dataset_path'], 'rb') as fin:
    dataset = pickle.load(fin)
with open(args['lookup_path'], 'rb') as fin:
    lookup = pickle.load(fin)

word_embeddings = np.load(args['embed_word_path'])  # (vocab_size, embedding_dim)

vocabs = list(lookup['word2idx'].keys())  # list of unique vocabularies
token2index_lookup = lookup['word2idx']

distinct_chars = sorted(list(set(itertools.chain.from_iterable(vocabs))))
num_chars = len(distinct_chars)  # 60
char2index_lookup = {c: i for i, c in enumerate(distinct_chars)}

vocab_lens = list(map(len, vocabs))
# from collections import Counter
# vocab_len_ctr = Counter(vocab_lens)
#
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots()
# ax.bar(vocab_len_ctr.keys(), vocab_len_ctr.values())
#
# ax.set_xlabel('length of word')
# ax.set_ylabel('count')
# ax.legend()
# plt.show()

allowed_max_vocab_len = 15  # Set allowed maximum vocabulary length
vocab_groups = [[] for _ in range(allowed_max_vocab_len)]
for vocab in vocabs:
    length = len(vocab)
    if length <= allowed_max_vocab_len:
        vocab_groups[length - 1].append(vocab)


def gen_batch(vocab_groups, char2index_lookup, token2index_lookup, word_embeddings, batch_size, shuffle=True):
    """Generate ranodm batches of words from different word groups (identified by different lengths of words)

    Args:
        vocab_groups (list[list[str]]): groups of words. Words in each group have same length
        char2index_lookup (dict): map a character to an integer index
        token2index_lookup (dict): map a word(token) to an integer index
        word_embeddings (numpy.ndarray): word embedding matrix with shape (vocab_size, embedding_size)
        batch_size (int):
        shuffle (bool):

    Returns:
        (generator) with shape [(B, VAR_word_len), (B, embedding_size)]

    Notes:
        Make the vocabularies in each batch have same character-length, then no need for padding

        Each iteration, a tuple composed of "character indices of batch-of-words" and "embedding of batch-of-words" will be return.
        Note that the group is selected randomly, so the length of "character indices" is variable.
    """

    num_groups = len(vocab_groups)  # number of differect word groups
    num_samps = [len(group) for group in vocab_groups]  # number of words in each word group

    # Make indices to get word from different groups
    indices = []
    for i in range(num_groups):
        inds = [i for i in range(num_samps[i])]
        if shuffle:
            np.random.shuffle(inds)
        indices.append(inds)

    pockets = [i for i in range(num_groups)]  # Indicate the groups that are not empty
    pointers = [0] * len(pockets)  # groups of indices to select words

    while True:
        g = np.random.choice(pockets)

        if pointers[g] + batch_size >= num_samps[g]:
            keys = np.array(indices[g][pointers[g]:])  # (B,)
            words = np.array(vocab_groups[g])[keys]

            # Find corresponding word index
            word_inds = np.array([token2index_lookup[word] for word in words])
            # Get corresponding word embeddings
            word_embds = word_embeddings[word_inds]

            # Indices of each character in each above words
            char_inds = []
            for word in words:
                char_inds.append([char2index_lookup[char] for char in word])

            yield np.array(char_inds), word_embds

            pockets.remove(g)  # run out of certain group, remove it from pockets
            if not pockets:  # run out all groups, exit
                break

        else:
            keys = np.array(indices[g][pointers[g]: pointers[g] + batch_size])
            words = np.array(vocab_groups[g])[keys]

            word_inds = np.array([token2index_lookup[word] for word in words])
            word_embds = word_embeddings[word_inds]

            char_inds = []
            for word in words:
                char_inds.append([char2index_lookup[char] for char in word])

            yield np.array(char_inds), word_embds

            pointers[g] += batch_size


if __name__ == '__main__':
    # Hyperparameters
    lr = 0.001
    batch_size = 100
    char_extended_size = 30
    h_size = CHAR_EMBEDDING_DIM // 2
    num_epochs = 100

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Model
    model = CharEmbedding(num_chars, char_extended_size, h_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    print('Start training character embeddings')
    model.train()

    for epoch in range(1, num_epochs + 1):
        for i, (x, y) in enumerate(gen_batch(vocab_groups, char2index_lookup, token2index_lookup,
                                             word_embeddings, batch_size)):

            x = torch.from_numpy(x).long().to(device)  # (B, VAR_len)
            y = torch.from_numpy(y).to(device)  # (B, embedding_size)

            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print('Ecpoh: {}, Batch: {}, Loss: {}'.format(epoch, i, loss.item()))

    print('Finish training character embeddings')

    print('Start extracting character embeddings')
    model.eval()

    char_embeddings = [[] for _ in range(len(vocabs))]
    for word, idx in token2index_lookup.items():
        chars = [char2index_lookup[c] for c in word]
        chars = torch.tensor(chars, dtype=torch.long, device=device).unsqueeze(0)  # (1, VAR_len)
        embedding = model(chars).squeeze().tolist()  # (char_embedding_size,)
        char_embeddings[idx] = embedding

    char_embeddings = np.array(char_embeddings)
    np.save(args['embed_char_path'], char_embeddings)
    print('Finish extracting character embeddings')
