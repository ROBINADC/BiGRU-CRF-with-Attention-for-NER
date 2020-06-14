# -*- coding: utf-8 -*-

"""
Modules for Gamma model
- MultiHeadAttention
- FeedForward
- CRF
Created on 2020/5/15 
"""

__author__ = "Yihang Wu"

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import build_attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout_rate=0.0, attention_type='scaled_dot'):
        super().__init__()

        assert model_dim % num_heads == 0, 'model_dim should be devided by num_heads'

        self.h_size = model_dim
        self.num_heads = num_heads
        self.head_h_size = model_dim // num_heads

        self.linear_q = nn.Linear(self.h_size, self.h_size)
        self.linear_k = nn.Linear(self.h_size, self.h_size)
        self.linear_v = nn.Linear(self.h_size, self.h_size)

        self.attention = build_attention(attention_type, q_dim=self.head_h_size, k_dim=self.head_h_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.lnorm = nn.LayerNorm(model_dim)

    def forward(self, q, k, v, attn_mask=None):
        batch_size = q.size(0)

        # Residual
        residual = q

        # Linear projection
        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        # Form multi heads
        q = q.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_q, D / h)
        k = k.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_k, D / h)
        v = v.view(self.num_heads * batch_size, -1, self.head_h_size)  # (h * B, T_v, D / h)

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(self.num_heads, 1, 1)  # (h * B, T_q, T_k)

        context, attention = self.attention(q, k, v, attn_mask=attn_mask)
        # context: (h * B, T_q, D_v) attention: (h * B, T_q, T_k)

        # Concatenate heads
        context = context.view(batch_size, -1, self.h_size)  # (B, T_q, D)

        # Dropout
        output = self.dropout(context)  # (B, T_q, D)

        # Residual connection and Layer Normalization
        output = self.lnorm(residual + output)  # (B, T_q, D)

        return output, attention


class FeedForward(nn.Module):

    def __init__(self, model_dim=512, hidden_dim=2048, dropout_rate=0.0):
        super().__init__()

        self.model_dim = model_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.linear1 = nn.Linear(self.model_dim, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.model_dim)
        self.norm = nn.LayerNorm(model_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        output = self.linear2(F.relu(self.linear1(x)))

        output = self.dropout(output)

        output = self.norm(output + x)
        return output


class CRF(nn.Module):

    def __init__(self, num_entities, pad_idx, bos_idx, eos_idx, device):
        super().__init__()
        self.num_entities = num_entities
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.cuda = device == 'cuda'

        self.transitions = nn.Parameter(torch.empty(self.num_entities, self.num_entities))  # treated as Parameter

        self._init_transitions()

    def forward(self, emissions, entities, mask=None):
        """Forward logic that computes the negative logarithm likelihood

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            entities (torch.LongTensor): given entities sequence (B, T)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (torch.Tensor): neg-log-likelihood as loss, mean over batch (1,)

        """

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float, device='cuda' if self.cuda else 'cpu').bool()

        score = self._score(emissions, entities, mask)  # (B,)
        partition = self._log_partition(emissions, mask)  # (B,)
        return -torch.mean(score - partition)  # (1,)

    def _init_transitions(self):
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        # Enforce contraints with a big negative number (from 'row' to 'column')
        # so exp(-10000) will tend to zero
        self.transitions.data[:, self.bos_idx] = -10000  # no transition to BOS
        self.transitions.data[self.eos_idx, :] = -10000  # no transition from eos (except to PAD)
        self.transitions.data[self.pad_idx, :] = -10000  # no transition from pad (except to PAD)
        self.transitions.data[:, self.pad_idx] = -10000  # no transition to pad (except from PAD)
        self.transitions.data[self.pad_idx, self.pad_idx] = 0
        self.transitions.data[self.pad_idx, self.eos_idx] = 0

    def _score(self, emissions, entities, mask):
        """

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            entities (torch.LongTensor): given entities sequence (B, T)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            torch.Tensor: scores of each entity sequence in current batch: (B,)

        """
        batch_size, seq_len = entities.shape

        scores = torch.zeros(batch_size, dtype=torch.float, device='cuda' if self.cuda else 'cpu')

        first_ents = entities[:, 0]  # first entities (B,)
        last_valid_idx = mask.sum(1) - 1  # (B,)
        last_ents = entities.gather(1, last_valid_idx.unsqueeze(1)).squeeze()  # last entities (B,)

        t_scores = self.transitions[self.bos_idx, first_ents]

        e_scores = emissions[:, 0].gather(1, first_ents.unsqueeze(1)).squeeze()  # (B,)

        scores += e_scores + t_scores

        # For each remaining entities(words)
        for i in range(1, seq_len):
            prev_ents = entities[:, i - 1]  # previous entities
            curr_ents = entities[:, i]  # current entities

            # Calculate emission and transition scores
            e_scores = emissions[:, i].gather(1, curr_ents.unsqueeze(1)).squeeze()  # (B,)
            t_scores = self.transitions[prev_ents, curr_ents]  # (B,)

            # Apply masking --- If position is PAD, then the contributing score should be 0
            e_scores = e_scores * mask[:, i]
            t_scores = t_scores * mask[:, i]

            scores += e_scores + t_scores

        # Transition score from last entity to EOS
        scores += self.transitions[last_ents, self.eos_idx]

        return scores  # (B,)

    def _log_partition(self, emissions, mask):
        """

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (torch.Tensor): partition scores for current batch (B,)
        """

        batch_size, seq_len, num_ents = emissions.shape

        # Antilog of partition part of score, which will be finally applied a logsumexp()
        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents)

        for i in range(1, seq_len):
            alpha_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1)  # (B, 1)

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0)  # (1, num_entities)

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas  # (B, num_entities)

                alpha_t.append(torch.logsumexp(scores, dim=1))  # inner (B,)

            new_alphas = torch.stack(alpha_t, dim=0).t()  # (B, num_entities)

            masking = mask[:, i].int().unsqueeze(1)  # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities)

        return torch.logsumexp(end_scores, dim=1)

    def viterbi_decode(self, emissions, mask):
        """Find the most probable entity sequence by applying viterbi decoding

        Args:
            emissions (torch.Tensor): emission matrix, which should be the output of previous layer (B, T, num_entities)
            mask (torch.BoolTensor): indicates valid positions within each sequence in the batch (B, T)

        Returns:
            (tuple): tuple containing:
                (torch.Tensor): viterbi score for each sequence in current batch (B,).
                (list[list[int]]): best sequences of entities of this batch, representing in indexes (B, *)

        """
        batch_size, seq_len, num_ents = emissions.shape

        alphas = self.transitions[self.bos_idx, :].unsqueeze(0) + emissions[:, 0]  # (B, num_ents)

        backpointers = []

        for i in range(1, seq_len):
            alpha_t = []
            backpointers_t = []

            for ent in range(num_ents):
                # Emission scores for current entity, broadcast to entity-wise
                e_scores = emissions[:, i, ent].unsqueeze(1)  # (B, 1)

                # Transition scores for current entity, broadcast to batch-wise
                t_scores = self.transitions[:, ent].unsqueeze(0)  # (1, num_entities)

                # Combine current scores with previous alphas (in log space: add instead of multiply)
                scores = e_scores + t_scores + alphas  # (B, num_entities)

                # Find the highest score and the entity associated with it
                max_scores, max_ents = torch.max(scores, dim=-1)

                # Add the max score for current entity
                alpha_t.append(max_scores)  # inner: (B,)

                # Add the corresponding entity to backpointers
                backpointers_t.append(max_ents)  # backpointers_t finally (num_entities, B)

            new_alphas = torch.stack(alpha_t, dim=0).t()  # (B, num_entities)

            masking = mask[:, i].int().unsqueeze(1)  # (B, 1)
            alphas = masking * new_alphas + (1 - masking) * alphas

            backpointers.append(backpointers_t)  # finally (T-1, num_entities, B)

        # Transition scores from last entity to EOS
        end_scores = alphas + self.transitions[:, self.eos_idx].unsqueeze(0)  # (B, num_entities)

        # Final most probable score and corresponding entity
        max_final_scores, max_final_ents = torch.max(end_scores, dim=-1)  # (B,)

        # Decode the best sequence for current batch
        # Follow the backpointers to find the best sequence of entities
        best_seqs = []
        emission_lens = mask.sum(dim=1)  # (B,)

        for i in range(batch_size):
            sample_len = emission_lens[i].item()

            sample_final_entity = max_final_ents[i].item()

            sample_backpointers = backpointers[: sample_len - 1]

            best_path = [sample_final_entity]

            best_entity = sample_final_entity

            for backpointers_t in reversed(sample_backpointers):
                best_entity = backpointers_t[best_entity][i].item()
                best_path.insert(0, best_entity)

            best_seqs.append(best_path)  # best_seqs finally (B, *)

        return max_final_scores, best_seqs
