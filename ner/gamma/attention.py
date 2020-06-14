# -*- coding: utf-8 -*-

"""
Different types of attention
Created on 2020/5/25 
"""

__author__ = "Yihang Wu"

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTENTION_REGISTRY = {}


def build_attention(attention_type, *args, **kwargs):
    return ATTENTION_REGISTRY[attention_type](*args, **kwargs)


def register_attention(name):
    """Decorator to register a new attention type"""

    def register_attention_cls(cls):
        if name in ATTENTION_REGISTRY:
            raise ValueError('Cannot register duplicate attention ({})'.format(name))
        if not issubclass(cls, BaseAttention):
            raise ValueError('Attention ({} : {}) must extend BaseAttention'.format(name, cls.__name__))

        ATTENTION_REGISTRY[name] = cls
        return cls

    return register_attention_cls


class BaseAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


@register_attention('dot')
class DotProductAttention(BaseAttention):
    """Dot Product Attention"""

    def __init__(self, dropout_rate=0.0, **kwargs):
        """Initialize DotProductAttention

        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)

        """
        attention = torch.bmm(q, k.permute(0, 2, 1))  # (B, T_q, T_k)

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention


@register_attention('scaled_dot')
class ScaledDotProductAttention(BaseAttention):
    """Scaled dot-product attention calculation"""

    def __init__(self, dropout_rate=0.0, **kwargs):
        """Initialize ScaledDotProductAttention

        Args:
            dropout_rate (float): attention dropout_rate rate
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)

        """
        attention = torch.bmm(q, k.permute(0, 2, 1))  # (B, T_q, T_k)

        # Scale
        attention *= k.size(-1) ** -0.5

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention


@register_attention('cosine')
class CosineAttention(BaseAttention):
    """Cosine Attention"""

    def __init__(self, dropout_rate=0.0, eps=1e-10, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.eps = eps

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)

        Notes:
            Consine attention requires D_q = D_k, so I denote it as D here

        """

        q_norm = q / (q.norm(p=2, dim=-1, keepdim=True) + self.eps)  # (B, T_q, D)
        k_norm = k / (k.norm(p=2, dim=-1, keepdim=True) + self.eps)  # (B, T_k, D)

        attention = torch.bmm(q_norm, k_norm.permute(0, 2, 1))  # (B, T_q, T_k)

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention


@register_attention('general')
class GeneralAttention(BaseAttention):
    """General Attention"""

    def __init__(self, q_dim, k_dim, dropout_rate=0.0):
        super().__init__()

        self.weights = nn.Parameter(torch.empty(q_dim, k_dim))
        self.dropout = nn.Dropout(dropout_rate)

        self._init_weights()

    def _init_weights(self):
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, q, k, v, attn_mask=None):
        """Forward

        Args:
            q (torch.Tensor): Query matrix, (B, T_q, D_q)
            k (torch.Tensor): Key matrix, (B, T_k, D_k)
            v (torch.Tensor): Value matrix, (B, T_v, D_v) T_v = T_k, D_v = D_k
            attn_mask (torch.BoolTensor | None): Mask tensor. True element will be masked.

        Returns:
            output (B, T_q, D_v); attention (B, T_q, T_k)
        """

        attention = q.matmul(self.weights).bmm(k.permute(0, 2, 1))  # (B, T_q, T_k)

        if attn_mask is not None:
            attention.masked_fill_(attn_mask, -np.inf)  # positions that require masking are now -np.inf

        attention = F.softmax(attention, dim=-1)

        attention = self.dropout(attention)

        output = attention.bmm(v)  # (B, T_q, D_v)

        return output, attention
