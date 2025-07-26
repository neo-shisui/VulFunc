import torch
import torch.nn as nn

# Define the LSTM model with optimizations
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers,
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

        # self.lstm = nn.LSTM(embedding_dim, 
        #                    hidden_dim, 
        #                    num_layers=num_layers,
        #                    batch_first=True, 
        #                    dropout=dropout if num_layers > 1 else 0)
        # self.dropout = nn.Dropout(dropout)
        # # Multiply by 2 for bidirectional
        # self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x: (batch_size, seq_len, embedding_dim)
        x, (h_n, c_n) = self.lstm(x)
        # x: (batch_size, seq_len, hidden_dim)
        x = self.dropout(x[:, -1, :])  # Use the last hidden state
        # x: (batch_size, hidden_dim)
        x = self.fc(x)
        # x: (batch_size, num_classes)
        return x
    
# Example usage
if __name__ == "__main__":
    vocab_size = 30000
    embedding_dim = 128
    hidden_dim = 256
    num_layers = 2
    num_classes = 10
    dropout = 0.2

    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout)

    dummy_input = torch.randint(0, vocab_size, (2, 10))  # batch of 2 sequences, each of length 10
    output = model(dummy_input)
    print("Output shape:", output.shape)  # Expected: (2, num_classes)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))  # Print number of trainable parameters
    print("Model structure:", model)  # Print the model structure
    # This will print the model structure and the number of trainable parameters
    # to verify the model's architecture and complexity
    # Note: The output shape should match the number of classes specified
    # and the model should be able to handle variable-length sequences.