# https://medium.com/@wangdk93/lstm-from-scratch-c8b4baf06a8b

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def pre_pad_sequences_pytorch(sequences, max_len, padding_token='0'):
    padded_sequences = []
    
    for seq in sequences:
        # If the sequence is shorter than max_len, pad with zeros at the beginning
        if len(seq) < max_len:
            padded_seq = seq + [padding_token] * (max_len - len(seq))  # Pre-padding with 0
        # If the sequence is longer than max_len, truncate it
        else:
            padded_seq = seq[-max_len:]  
        padded_sequences.append(padded_seq)
    
    return torch.tensor(padded_sequences)

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
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        x = self.embedding(x)
        # x: (batch_size, seq_len, embedding_dim)
        x, (h_n, c_n) = self.lstm(x)
        # x: (batch_size, seq_len, hidden_dim)
        x = self.dropout(h_n[-1])  # Last layer's hidden state 
        # self.dropout(x[:, -1, :])  # Use the last hidden state
        # x: (batch_size, hidden_dim)
        x = self.fc(x)
        # x: (batch_size, num_classes)
        return x

# Training function
def train_lstm(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, num_classes=2):
    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        total_confusion = torch.zeros((num_classes, num_classes), device=device)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                print(outputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                # Compute batch confusion matrix
                indices = labels * num_classes + predicted
                batch_confusion = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                total_confusion += batch_confusion
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # Calculate metrics from confusion matrix
        tp = total_confusion.diag()
        fp = total_confusion.sum(0) - tp
        fn = total_confusion.sum(1) - tp
        precision_per_class = tp / (tp + fp + 1e-6)  # Add small epsilon to avoid division by zero
        recall_per_class = tp / (tp + fn + 1e-6)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-6)
        accuracy = tp.sum() / total_confusion.sum()
        precision = precision_per_class.mean()  # Macro-average
        recall = recall_per_class.mean()
        f1 = f1_per_class.mean()

        # Store metrics
        test_accuracies.append(accuracy.item())
        test_precisions.append(precision.item())
        test_recalls.append(recall.item())
        test_f1s.append(f1.item())

        # Print all metrics for the epoch
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, '
              f'Test Recall: {recall:.4f}, Test F1: {f1:.4f}')

    return train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s

# Function to plot training metrics
def lstm_plot_metrics(train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 8))

    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, test_losses, 'r-', label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot test metrics
    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_accuracies, 'g-', label='Test Accuracy')
    plt.plot(epochs, test_precisions, 'm-', label='Test Precision')
    plt.plot(epochs, test_recalls, 'c-', label='Test Recall')
    plt.plot(epochs, test_f1s, 'y-', label='Test F1')
    plt.title('Test Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('lstm.png')  # Save the plot (replace 'plt.save()' with correct syntax)
    plt.show()

# Example usage
if __name__ == "__main__":
    vocab_size = 30000
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    num_classes = 2
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