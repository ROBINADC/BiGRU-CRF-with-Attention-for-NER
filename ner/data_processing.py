# -*- coding: utf-8 -*-

"""
Data Processing
Created on 2020/5/13 
"""

__author__ = "Yihang Wu"

import pickle
import itertools

from arguments import Arguments as arg
from gamma import DataProcessing

if __name__ == '__main__':
    # Read original csv files
    sens_train, ents_train = DataProcessing.read_csv(arg.raw_data_train)
    sens_val, ents_val = DataProcessing.read_csv(arg.raw_data_val)
    sens_test, _ = DataProcessing.read_csv(arg.raw_data_test, is_test=True)

    # Build Word-to-Index lookup table
    word_list = list(set(itertools.chain.from_iterable(sens_train + sens_val + sens_test)))
    extra_sign_dict = {sign: idx for sign, idx in [arg.word_pad, arg.word_oov]}
    word2idx = DataProcessing.build_lookup(word_list, **extra_sign_dict)

    # Build Entity-to-Index lookup table
    entity_list = list(set(itertools.chain.from_iterable(ents_train + ents_val)))
    extra_sign_dict = {sign: idx for sign, idx in [arg.entity_pad, arg.entity_bos, arg.entity_eos]}
    entity2idx = DataProcessing.build_lookup(entity_list, **extra_sign_dict)

    # Convert words and name entities to integer index
    sens_train = [[word2idx[w] for w in sentence] for sentence in sens_train]
    sens_val = [[word2idx[w] for w in sentence] for sentence in sens_val]
    sens_test = [[word2idx[w] for w in sentence] for sentence in sens_test]
    ents_train = [[entity2idx[e] for e in ents] for ents in ents_train]
    ents_val = [[entity2idx[e] for e in ents] for ents in ents_val]

    # Pad sequences
    train_seq_len = max(len(sen) for sen in sens_train)
    sens_train_pad = DataProcessing.pad_sequence(sens_train, train_seq_len, word2idx[arg.word_pad[0]])
    ents_train_pad = DataProcessing.pad_sequence(ents_train, train_seq_len, entity2idx[arg.entity_pad[0]])

    val_seq_len = max(len(sen) for sen in sens_val)
    sens_val_pad = DataProcessing.pad_sequence(sens_val, val_seq_len, word2idx[arg.word_pad[0]])
    ents_val_pad = DataProcessing.pad_sequence(ents_val, val_seq_len, entity2idx[arg.entity_pad[0]])

    test_seq_len = max(len(sen) for sen in sens_test)
    sens_test_pad = DataProcessing.pad_sequence(sens_test, test_seq_len, word2idx[arg.word_pad[0]])

    # Store relevant data to file
    lookup = dict()
    lookup['word2idx'] = word2idx
    lookup['entity2idx'] = entity2idx

    with open(arg.lookup_path, 'wb') as fout:
        pickle.dump(lookup, fout)

    dataset = dict()
    dataset['sens_train'] = sens_train
    dataset['sens_val'] = sens_val
    dataset['sens_test'] = sens_test
    dataset['ents_train'] = ents_train
    dataset['ents_val'] = ents_val

    with open(arg.dataset_path, 'wb') as fout:
        pickle.dump(dataset, fout)

    padded = dict()
    padded['sens_train'] = sens_train_pad
    padded['ents_train'] = ents_train_pad
    padded['sens_val'] = sens_val_pad
    padded['ents_val'] = ents_val_pad
    padded['sens_test'] = sens_test_pad

    with open(arg.padded_dataset_path, 'wb') as fout:
        pickle.dump(padded, fout)
