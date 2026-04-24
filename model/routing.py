import torch
import torch.nn as nn

class RoutingTokens(nn.Module):
    def __init__(self, num_tokens, d_model):
        super().__init__()
        self.tokens = nn.Parameter(torch.randn(num_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, 1, batch_first=True)

    def forward(self, x):
        B, T, D = x.shape

        routing = self.tokens.unsqueeze(0).expand(B, -1, -1)

        routing_out, _ = self.attn(routing, x, x)
        x_out, _ = self.attn(x, routing_out, routing_out)

        return x_out