import torch.nn as nn
from model.local_mixer import LocalSelfAttention
from model.routing import RoutingTokens

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size,
                 ff_dim, num_routing_tokens, dropout):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.local_attn = LocalSelfAttention(
            d_model, num_heads, window_size, dropout
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.routing = RoutingTokens(num_routing_tokens, d_model)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.local_attn(self.norm1(x))
        x = x + self.routing(self.norm2(x))
        x = x + self.ff(self.norm3(x))
        return x