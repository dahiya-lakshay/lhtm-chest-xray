import torch.nn as nn
from utils.mask_utils import local_window_mask

class LocalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.window_size = window_size

    def forward(self, x):
        B, T, D = x.shape
        mask = local_window_mask(T, self.window_size, x.device)
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out

#interest
