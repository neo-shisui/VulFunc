import torch
import torch.nn as nn

class CXXTokenEmbedding(nn.Module):
    """
    Embedding layer for tokenized C++ code.
    """
    def __init__(self, vocab_size, embed_dim, padding_idx=0):
        super(CXXTokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)

    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        # returns: (batch_size, seq_len, embed_dim)
        return self.embedding(input_ids)

# Example usage
if __name__ == "__main__":
    vocab_size = 30000
    embed_dim = 128
    model = CXXTokenEmbedding(vocab_size, embed_dim)

    dummy_input = torch.randint(0, vocab_size, (2, 10))  # batch of 2 sequences, each of length 10
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, 10, 128)
