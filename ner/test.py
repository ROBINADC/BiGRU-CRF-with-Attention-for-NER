# -*- coding: utf-8 -*-

"""
Specialized test program
Dummy target entities are created since the entities in test data are not provided
Created on 2020/5/28 
"""

__author__ = "Yihang Wu"

import os
import csv
import pickle

import torch

from gamma import build_model, cal_f1score
from arguments import GRUAttnCRFArguments as arg


def run(arg):
    with open(arg.dataset_path, 'rb') as fin:
        dataset = pickle.load(fin)

    with open(arg.lookup_path, 'rb') as fin:
        lookup = pickle.load(fin)

    word2idx = lookup['word2idx']
    entity2idx = lookup['entity2idx']
    idx2entity = {idx: ent for ent, idx in entity2idx.items()}
    o_entity = entity2idx['O']

    test_sens = dataset['sens_test']  # CHECK
    test_ents = [[o_entity for _ in sen] for sen in test_sens]  # create dummpy entities

    arg.num_vocabs = len(word2idx)
    arg.num_entities = len(entity2idx)

    model = build_model(arg.model_name, arg).to(arg.device)

    # Load existed weights
    if os.path.exists(arg.test_ckpt):
        model.load_state_dict(torch.load(arg.test_ckpt))

    y_true, y_pred = [], []

    model.eval()
    for sentence, entity in zip(test_sens, test_ents):
        x = torch.tensor(sentence, dtype=torch.long).unsqueeze(0).to(arg.device)
        y = torch.tensor(entity, dtype=torch.long).unsqueeze(0).to(arg.device)
        _, preds = model(x, y)

        y_true.append(entity)
        y_pred.append(preds[0])

    y_true_flatten = sum(y_true, [])
    y_pred_flatten = sum(y_pred, [])
    # print(cal_f1score(y_true_flatten, y_pred_flatten))

    # Write to csv
    if arg.write_to_csv:
        if not os.path.exists(arg.csv_dir):
            os.makedirs(arg.csv_dir)

        with open(os.path.join(arg.csv_dir, 'output.csv'), 'w', newline='') as cout:
            writer = csv.writer(cout)
            writer.writerow(('Id', 'Predicted'))
            writer.writerows(enumerate(idx2entity[idx] for idx in y_pred_flatten))


if __name__ == '__main__':
    run(arg)
