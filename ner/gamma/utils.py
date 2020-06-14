# -*- coding: utf-8 -*-

"""
Utilities of Gamma model
Created on 2020/5/15 
"""

__author__ = "Yihang Wu"

import numpy as np
import torch
from sklearn.metrics import f1_score


def load_embeddings(*fps):
    """Load embeddings for inputs, concatenate embeddings if multiple files provided

    Args:
        *fps (Iterable[str]): single or multiple existed embedding npy files

    Returns:
        (numpy.ndarray): concatenated embeddings
    """
    embeddings = [np.load(fp) for fp in fps]
    embedding = np.concatenate(embeddings, axis=-1)
    return embedding


def attention_padding_mask(q, k, padding_index=0):
    """Generate mask tensor for padding value

    Args:
        q (Tensor): (B, T_q)
        k (Tensor): (B, T_k)
        padding_index (int): padding index. Default: 0

    Returns:
        (torch.BoolTensor): Mask with shape (B, T_q, T_k). True element stands for requiring making.

    Notes:
        Assume padding_index is 0:
        k.eq(0) -> BoolTensor (B, T_k)
        k.eq(0).unsqueeze(1)  -> (B, 1, T_k)
        k.eq(0).unsqueeze(1).expand(-1, q.size(-1), -1) -> (B, T_q, T_k)

    """

    mask = k.eq(padding_index).unsqueeze(1).expand(-1, q.size(-1), -1)
    return mask


def cal_accuracy(predicts, targets, ignore_index=None):
    """Calculate accuracy simply

    Args:
        predicts (numpy.ndarray): predicted indexes
        targets (numpy.ndarray): gold true indexes
        ignore_index (int | None): index in targets that needs to be negleted

    Returns:
        (float): Accuracy

    """
    assert predicts.shape == targets.shape, 'predicts and targets should have same shape'

    if ignore_index is not None:
        valid = targets != ignore_index
        predicts = predicts[valid]  # would be flattened and with valid positions chosen (*)
        targets = targets[valid]  # would be flattened and with valid positions chosen (*)

    return np.sum(predicts == targets) / predicts.size


def cal_f1score(y_true, y_pred):
    """Calculate mean f1 score by using scikit-learn package

    Args:
        y_true (list): list of true tags
        y_pred (list): list of predicted tags

    Returns:
        (int): mean f1 score
    """
    return f1_score(y_true, y_pred, average='micro')


def pad_seq(sequences, max_len=None, batch_first=True, padding_value=0):
    """Pad a list of variable length ndarrays with padding_value

    Args:
        sequences (list[numpy.ndarray]): stacks a list of ndarray along a new dimension
        max_len (int): maximum length of sequence. If None, then auto calculate.
        batch_first (bool): if the first dimension of return data indicating batch
        padding_value (int): padding value

    Returns:
        (numpy.ndarray): padded arrays with shape (B, max_len, *)
    """

    # assuming trailing dimensions and type of all the ndarrays
    # in sequences are same and fetching those from sequences[0]
    sample_size = sequences[0].shape
    trailing_dims = sample_size[1:]

    max_len = max_len if max_len is not None else max([s.shape[0] for s in sequences])

    if batch_first:
        out_dims = (len(sequences), max_len) + trailing_dims
    else:
        out_dims = (max_len, len(sequences)) + trailing_dims

    out_array = np.full(out_dims, fill_value=padding_value, dtype=sequences[0].dtype)

    for i, seq in enumerate(sequences):
        seq_len = len(seq)
        if batch_first:
            out_array[i, :seq_len, ...] = seq
        else:
            out_array[:seq_len, i, ...] = seq

    return out_array


def decode_entity(x, mask):
    """Decode sequences of entities from weight matrix

    Args:
        x (torch.Tensor): output with shape (B, T, num_entities)
        mask (torch.BoolTensor): (B, T)

    Returns:
        (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)
    """
    first_invalid = mask.sum(1)  # (B,)

    preds = x.argmax(dim=-1)  # (B, T)
    path = [preds[i].data[:first_invalid[i].item()].tolist() for i in range(preds.shape[0])]  # (B, *)
    return path


class EarlyStopping:
    def __init__(self, monitor='loss', min_delta=0., patience=0):
        """EarlyStopping

        Args:
            monitor (str): quantity to be monitored. 'loss' or 'acc'
            min_delta (float): minimum change in the monitored quantity to qualify as an improvement
                i.e. an absolute change of less than min_delta, will count as no improvement.
            patience (int): number of epochs with no improvement after which training will be stopped
        """

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience

        self._wait = 0
        self._best = None
        self._best_epoch = None

        if 'loss' in self.monitor:
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1

    def judge(self, epoch, value):
        current = value

        if self._best is None:
            self._best = current
            self._best_epoch = epoch
            return

        if self.monitor_op(current - self.min_delta, self._best):
            self._best = current
            self._best_epoch = epoch
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                return True

    @property
    def best_epoch(self):
        return self._best_epoch

    @property
    def best_val(self):
        return self._best
