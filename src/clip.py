import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Parameter(torch.zeros(torch.zeros(n_tokens, embed_size)))

    def forward(self, tokens):
        x = self.token_embedding(tokens)

        x += self.position_embedding(tokens)

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, embed_size: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(n_head, embed_size)
        self.layernorm_2 = nn.LayerNorm(embed_size)
        self.linear_1 = nn.Linear(embed_size, 4 * embed_size)
        self.linear_2 = nn.Linear(4 * embed_size, n_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Self Attention
        x = self.layernorm_1(x)
        x = self.attention(x, causal_mask=True)

        x += residual

        ## FFN

        residual = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = nn.GeLU(x) # x * torch.sigmoid(1.702 * x): Quick GeLU activation function

        x = self.linear_2(x)

        x += residual

        return x

class CLIP(nn.Module):
    def __init__(self):
        self.embedding = CLIPEmbedding(vocab_size=49408, edmbed_size = 768, n_tokens=77)
        self.layers = nn.Module(
            CLIPLayer(12, 768) for i in range(12)
        )

        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # [B, seq_len] -> [B, seq_len, dim]
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        [B, seq_len, dim]
        output = self.layernorm(state)

        return output