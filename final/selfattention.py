import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class SelfAttentionModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.2):
        super(SelfAttentionModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Attention parameters
        self.query = nn.Linear(embedding_dim, hidden_dim)
        self.key = nn.Linear(embedding_dim, hidden_dim)
        self.value = nn.Linear(embedding_dim, hidden_dim)
        self.scale = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # embedded shape: (batch_size, seq_length, embedding_dim)
        
        # Compute query, key, value
        Q = self.query(embedded)  # (batch_size, seq_length, hidden_dim)
        K = self.key(embedded)    # (batch_size, seq_length, hidden_dim)
        V = self.value(embedded)  # (batch_size, seq_length, hidden_dim)
        
        # Compute attention scores
        attention_scores = torch.bmm(Q, K.transpose(1, 2)) / self.scale
        # attention_scores shape: (batch_size, seq_length, seq_length)
        
        # Apply softmax to get attention weights
        attention_weights = torch.nn.functional.softmax(attention_scores, dim=-1)
        # attention_weights shape: (batch_size, seq_length, seq_length)
        
        # Apply attention weights to values
        attention_output = torch.bmm(attention_weights, V)
        # attention_output shape: (batch_size, seq_length, hidden_dim)
        
        # Pool the output (using mean pooling across sequence length)
        pooled_output = torch.mean(attention_output, dim=1)
        # pooled_output shape: (batch_size, hidden_dim)
        
        # Apply dropout
        out = self.dropout(pooled_output)
        
        # Final linear layer
        out = self.fc(out)
        # out shape: (batch_size, num_classes)
        
        return out

def train_selfattention(model, train_loader, optimizer, scheduler, epochs, device, criterion):
    model.train()
    scaler = torch.amp.GradScaler('cuda')  # For mixed precision training

    history = {"loss": [], "accuracy": []}

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
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
        history["loss"].append(avg_loss)
        history["accuracy"].append(accuracy)

    return model, history