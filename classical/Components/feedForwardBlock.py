import torch
from torch import nn

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(in_features = d_model, out_features = d_ff) # W1 & B1
        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(in_features = d_ff, out_features = d_model) # W2 & B2

    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))