# https://medium.com/@wangdk93/lstm-from-scratch-c8b4baf06a8b

from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define the LSTM model with optimizations
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(LSTMModel, self).__init__()
        # Define Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=num_layers,
                            batch_first=True, 
                            dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x       : (batch_size, seq_len)
        # embedded: (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)

        packed_output, (h_n, c_n) = self.lstm(embedded)

        # x = self.dropout(h_n[-1])  # Last layer's hidden state 
        out = self.dropout(packed_output[:, -1, :])  # Use last time step's output
        # self.dropout(x[:, -1, :])  # Use the last hidden state
        # x: (batch_size, hidden_dim)
        out = self.fc(out)
        # x: (batch_size, num_classes)
        return out

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

        # Print predict output (1 or 0) and labels for debugging
        # print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}')
        # print("Sample Outputs:", outputs[:5])  # Print first 5 outputs
        # print("Sample Labels:", labels[:5])

        print("Raw logits:", outputs[:5])  # Should show varying values for both classes
        print("Predicted classes:", torch.max(outputs, 1)[1][:5])  # Should show both 0 and 1
        print("True labels:", labels[:5])  # Should show both 0 and 1

        # Evaluation phase
        model.eval()
        test_loss = 0.0
        total_confusion = torch.zeros((num_classes, num_classes), device=device)
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                # print(outputs)
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

# Define the BiLSTM model with optimizations
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.2):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=num_layers, 
                           bidirectional=True, 
                           batch_first=True, 
                           dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        # Multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        # Pack padded sequence for faster processing
        # This helps with variable length sequences
        packed_output, (hidden, cell) = self.lstm(embedded)
        
        # Use the concatenated hidden state from the last layer
        # Get the last hidden state of both directions
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        hidden_cat = torch.cat((hidden_forward, hidden_backward), dim=1)
        
        # Apply dropout
        out = self.dropout(hidden_cat)
        
        # Linear layer
        out = self.fc(out)
        # out shape: (batch_size, num_classes)
        
        return out

def train_bilstm(model, train_loader, test_loader, optimizer, scheduler, epochs, device, criterion):
    if len(train_loader) == 0:
        raise ValueError("train_loader is empty")
    if len(test_loader) == 0:
        raise ValueError("test_loader is empty")
    
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    history = {"train_loss": [], "train_accuracy": [], "test_loss": [], "test_accuracy": []}

    # Training loop with early stopping
    for epoch in range(epochs):
        # Training loop
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})

        train_loss = total_loss / len(train_loader)
        train_accuracy = 100. * correct / total
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        # Test loop
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                with torch.amp.autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100. * test_correct / test_total
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)

        scheduler.step()
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

        # Early stopping condition
        # if epoch > 0 and test_loss > history["test_loss"][-2]:
        #     print("Early stopping triggered")
        #     break

    return model, history

def train_bilstm_old(model, train_loader, optimizer, scheduler, epochs, device, criterion):
    model.train()
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision training

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        history = {"loss": [], "accuracy": []}  # Thêm dòng này

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')

        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass with mixed precision
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            # Backward pass and optimize with gradient scaling
            scaler.scale(loss).backward()
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Track statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * correct / total
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.2f}%'
            })

        # Update learning rate based on scheduler
        scheduler.step()

        print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # logger.info(
        #     f'Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)


    return model, history

# Function to plot training metrics
def bilstm_plot_metrics(train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s):
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
    plt.savefig('bilstm_metrics.png')
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