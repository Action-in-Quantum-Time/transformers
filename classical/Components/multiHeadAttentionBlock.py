import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, 'd_model is not divisible by h'

        self.d_k = d_model // h
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)
        self.w_o = nn.Linear(in_features=d_model, out_features=d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout:nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / np.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)   # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_k)
        query = query.view(query.shapr[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = query.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    
class AttentionBlock(nn.Module):
    def __init__(self, seq_len=200):
        super().__init__()
        self.queries = nn.Linear(seq_len, seq_len)
        self.keys = nn.Linear(seq_len, seq_len)
        self.values = nn.Linear(seq_len, seq_len)

    def forward(self, x, mask=True):
        q = self.queries(x).reshape(x.shape[0], x.shape[1], 1)
        k = self.keys(x).reshape(x.shape[0], x.shape[1], 1)
        v = self.values(x).reshape(x.shape[0], x.shape[1], 1)
        scores = torch.bmm(q, k.transpose(-2, -1))
        if mask:
            maskmat = torch.tril(torch.ones((x.shape[1], x.shape[1])))
            scores = scores.masked_fill(maskmat == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.bmm(attention_weights, v)
        return output.reshape(output.shape[0], output.shape[1])    