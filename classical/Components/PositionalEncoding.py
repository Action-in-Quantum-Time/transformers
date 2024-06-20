import numpy as np
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
    
        # Matrix of shape(seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Vector of shape (seq_len)
        position = torch.arange(0, seq_len,dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # Apply sin to even position
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # It will have the shape (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad(False)
        return self.dropout(x)