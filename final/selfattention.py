# Credit:
# * https://medium.com/@devmallyakarar/transformers-self-attention-mechanism-from-scratch-using-pytorch-affee86f9ba9

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        
        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]  # Number of sequences (batch size)
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        # Dot product attention
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))
        
        # Scaled dot-product attention
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        
        out = self.fc_out(out)
        
        return out

# Transformer Model incorporating SelfAttention
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, heads, num_layers, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_size),
                SelfAttention(embed_size, heads),
                nn.Dropout(dropout),
                nn.Linear(embed_size, embed_size),
                nn.ReLU(),
                nn.LayerNorm(embed_size)
            ) for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        out = self.fc_out(x)
        return out

# Example usage
if __name__ == "__main__":
    embed_size = 256
    heads = 8
    seq_length = 10
    x = torch.rand((1, seq_length, embed_size))  # Example input

    self_attention = SelfAttention(embed_size, heads)
    out = self_attention(x, x, x, mask=None)
    print(out.shape)  # Output shape should be [1, seq_length, embed_size]