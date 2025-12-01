import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask = False):
        # x: [B, seq_len, dim]
        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape

        hidden_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # [B, seq_len, dim] -> [B, seq_len, dim * 3] -> 3 tensors of shape [B, seq_len, dim]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # [B, seq_len, dim] -> [B, seq_len, n_heads, d_head] -> [B, n_heads, seq_len, d_head]
        q = q.view(hidden_shape).transpose(1, 2)
        k = k.view(hidden_shape).transpose(1, 2)
        v = v.view(hidden_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # [B, n_heads, seq_len, seq_len] @ [B, n_heads, seq_len, d_head] -> [B, n_heads, seq_len, d_head]
        output = weight @ v

        # [B, n_heads, seq_len, d_head] -> [B, seq_len, n_head, d_head]
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)
        output = self.out_proj(output)

        return output

