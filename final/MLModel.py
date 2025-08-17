import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Training function for DecisionTree
def train_decision_tree(model, X_train, y_train, X_test, y_test, num_epochs, feature_names=None):
    # Initialize lists to store metrics
    train_accuracies = []
    test_accuracies = []
    test_precisions = []
    test_recalls = []
    test_f1s = []

    # DecisionTree doesn't use epochs, but we simulate for consistency
    for epoch in range(num_epochs):
        # Training phase
        model.fit(X_train, y_train)
        
        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Compute metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, average='macro', zero_division=0)
        test_recall = recall_score(y_test, test_pred, average='macro', zero_division=0)
        test_f1 = f1_score(y_test, test_pred, average='macro', zero_division=0)

        # Store metrics
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        test_precisions.append(test_precision)
        test_recalls.append(test_recall)
        test_f1s.append(test_f1)

        # Print TP, FP, FN for debugging
        print(f'True Positives: {np.sum((y_test == 1) & (test_pred == 1))}')
        print(f'False Positives: {np.sum((y_test == 0) & (test_pred == 1))}')
        print(f'False Negatives: {np.sum((y_test == 1) & (test_pred == 0))}')
        print(f'True Negatives: {np.sum((y_test == 0) & (test_pred == 0))}')

        # Print metrics
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Accuracy: {train_accuracy:.4f}, '
              f'Test Accuracy: {test_accuracy:.4f}, '
              f'Test Precision: {test_precision:.4f}, '
              f'Test Recall: {test_recall:.4f}, '
              f'Test F1: {test_f1:.4f}')

        # Print sample predictions for debugging
        print("Sample Predicted classes:", test_pred[:5])
        print("Sample True labels:", y_test[:5])

    return train_accuracies, test_accuracies, test_precisions, test_recalls, test_f1s

# Function to plot training metrics
def decision_tree_plot_metrics(train_accuracies, test_accuracies, test_precisions, test_recalls, test_f1s):
    epochs = range(1, len(train_accuracies) + 1)
    plt.figure(figsize=(12, 8))

    # Plot accuracies
    plt.subplot(2, 1, 1)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
    plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot test metrics
    plt.subplot(2, 1, 2)
    plt.plot(epochs, test_precisions, 'm-', label='Test Precision')
    plt.plot(epochs, test_recalls, 'c-', label='Test Recall')
    plt.plot(epochs, test_f1s, 'y-', label='Test F1')
    plt.title('Test Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()

    plt.tight_layout()
    plt.savefig('decision_tree_metrics.png')
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load labels from your JSON file
    labels_df = pd.read_json('CXX_AST_DATA/labels.json', orient='records', lines=True)
    y = labels_df['target'].values

    # Simulated feature extraction from token lists
    # Replace this with your actual feature extraction (e.g., Bag-of-Words, TF-IDF)
    token_lists = [['translation_unit', 'function_definition'], ['translation_unit', 'function_definition2']]
    X = np.random.rand(len(y), 10)  # Simulated features (replace with real features)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)

    # Train and evaluate
    num_epochs = 5  # Simulate epochs for consistency
    train_accuracies, test_accuracies, test_precisions, test_recalls, test_f1s = train_decision_tree(
        model, X_train, y_train, X_test, y_test, num_epochs
    )
    decision_tree_plot_metrics(train_accuracies, test_accuracies, test_precisions, test_recalls, test_f1s)