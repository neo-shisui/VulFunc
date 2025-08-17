import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, 
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
        x, h_n = self.rnn(x)
        # x: (batch_size, seq_len, hidden_dim)
        x = self.dropout(x[:, -1, :])  # Use last time step's output
        # x: (batch_size, hidden_dim)
        x = self.fc(x)
        # x: (batch_size, num_classes)
        return x

# Training function
def train_rnn(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, num_classes=2, patience=5):
    # Initialize lists to store metrics
    train_losses = []
    test_losses = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    best_loss = float('inf')
    epochs_no_improve = 0

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
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                indices = labels * num_classes + predicted
                batch_confusion = torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
                total_confusion += batch_confusion
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        # Calculate metrics from confusion matrix
        tp = total_confusion.diag()
        fp = total_confusion.sum(0) - tp
        fn = total_confusion.sum(1) - tp
        precision_per_class = tp / (tp + fp + 1e-6)
        recall_per_class = tp / (tp + fn + 1e-6)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-6)
        accuracy = tp.sum() / total_confusion.sum()
        precision = precision_per_class.mean()
        recall = recall_per_class.mean()
        f1 = f1_per_class.mean()

        test_accuracies.append(accuracy.item())
        test_precisions.append(precision.item())
        test_recalls.append(recall.item())
        test_f1s.append(f1.item())

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {accuracy:.4f}, Test Precision: {precision:.4f}, '
              f'Test Recall: {recall:.4f}, Test F1: {f1:.4f}')

        # Early stopping
        if test_loss < best_loss:
            best_loss = test_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    return train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s

# Function to plot training metrics
def rnn_plot_metrics(train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s):
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
    plt.savefig('rnn_metrics.png')  # Save the plot
    plt.show()

# Example usage
if __name__ == "__main__":
    # Hyperparameters
    vocab_size = 30000  # Adjust based on your token vocabulary
    embedding_dim = 256
    hidden_dim = 256
    num_layers = 2
    num_classes = 2  # For binary classification (e.g., 0 or 1 from labels.json)
    dropout = 0.2

    # Initialize model
    model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Test forward pass
    dummy_input = torch.randint(0, vocab_size, (2, 10))  # Batch of 2 sequences, each of length 10
    output = model(dummy_input.to(device))
    print("Output shape:", output.shape)  # Expected: (2, num_classes)
    print("Model parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model structure:", model)