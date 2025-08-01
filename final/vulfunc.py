import argparse
import copy
import pandas as pd
import numpy as np
import os
import re
import json
import pickle
import keyword
import logging
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import tree_sitter_cpp
from tree_sitter import Language, Parser
from sklearn.model_selection import train_test_split

from cxx_node_tokenizer import CXXNodeTokenizer
from cxx_vocabulary_manager import CXXVocabularyManager
from cxx_normalization import CXXNormalization
from cxx_token_embedding import CXXTokenEmbedding
from lstm import LSTMModel, train_lstm, lstm_plot_metrics
from cxx_ast_traversal import get_root_paths, get_sequences
from selfattention import SelfAttention, TransformerModel

# Configurations
DATASET_PATH = [
    '../datasets/Devign/devign.pkl',
    '../datasets/SARD/sard.pkl',
]

VOCAB_PATH = 'CXX_AST_DATA/vocab/vocab.json'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='vulfunc.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def filter_tokens_by_vocab(tokens, vocab):
    """
    Filter tokens based on vocabulary and add UNK for unknown tokens.
    """
    result = []
    for token in tokens:
        if token in vocab:
            result.append(token)
        else:
            result.append('<unk>')
    return result

def convert_tokens_to_ids(tokenized_data, vocab):
    """
    Convert list of token lists into list of ID lists based on vocabulary.
    """
    token_ids_data = []
    for tokens in tokenized_data:
        token_ids = [vocab[token] if token in vocab else vocab['<unk>']
                     for token in tokens]
        token_ids_data.append(token_ids)
    return token_ids_data

def prepare_data(token_ids_data, labels, vocab, max_len=128, test_size=0.2, batch_size=32):
    """
    Prepare data for training by padding sequences, splitting into train/test sets,
    and creating DataLoader objects.

    Args:
        token_ids_data: List of lists containing token IDs
        labels: List of labels (0 for valid, 1 for anomalous)
        max_len: Maximum sequence length for padding
        test_size: Proportion of data to use for testing
        batch_size: Batch size for training

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for testing data
    """
    # Ensure token_ids_data and labels have the same length
    if len(token_ids_data) != len(labels):
        raise ValueError(
            f"Mismatch between token_ids_data length ({len(token_ids_data)}) and labels length ({len(labels)})")

    # Pad sequences to the same length
    padded_data = []
    for seq in token_ids_data:
        if len(seq) < max_len:
            padded_seq = seq + [vocab['<pad>']] * (max_len - len(seq))
        else:
            padded_seq = seq[:max_len]
        padded_data.append(padded_seq)

    # Convert to PyTorch tensors
    X = torch.tensor(padded_data, dtype=torch.long)
    # y = torch.tensor(labels, dtype=torch.long)
    y = torch.tensor(labels.values, dtype=torch.long)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    
    # Print unique value counts for y_train and y_test
    print("Training set (y_train) unique value counts:")
    unique, counts = np.unique(y_train, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"Class {value}: {count}")

    print("\nTest set (y_test) unique value counts:")
    unique, counts = np.unique(y_test, return_counts=True)
    for value, count in zip(unique, counts):
        print(f"Class {value}: {count}")


    # Create DataLoader objects
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def plot_history(history, title_prefix="", save_path=None):
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{title_prefix} Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["accuracy"], label="Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(f"{title_prefix} Accuracy")
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Training function
def train_model(model, dataloader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            inputs = batch.to(device)
            targets = inputs[:, 1:].contiguous()  # Shift for next-token prediction
            inputs = inputs[:, :-1]  # Exclude last token for input

            optimizer.zero_grad()
            outputs = model(inputs, mask=None)  # Mask can be added for causal attention
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

def standardize_entry(entry, source_idx):
    """Standardize a dictionary entry to a common format."""
    standardized = {}
    
    # Map code field
    if 'function' in entry:
        standardized['code'] = entry['function']
    elif 'code' in entry:
        standardized['code'] = entry['code']
    else:
        raise KeyError(f"Code field not found in entry from source {source_idx}")
    
    # Map label field
    if 'target' in entry:
        standardized['target'] = int(entry['label'])
    elif 'vulnerable' in entry:
        standardized['target'] = 1 if entry['vulnerable'] else 0
    else:
        standardized['target'] = None  # Or raise an error if label is required
    
    # Map path field
    if 'cwd' in entry:
        standardized['cwd'] = entry['path']
    elif 'filename' in entry:
        standardized['cwd'] = entry['filename']
    else:
        standardized['cwd'] = None
    
    return standardized

def generate_vocab():
    """
    Generate vocabulary from the dataset.
    """
    print("[+] Geneting vocab")
    # Load the dataset
    devign_dataset = pd.read_pickle(DATASET_PATH[0])
    sard_dataset = pd.read_pickle(DATASET_PATH[1])
    # print(sard_dataset.columns)
    # print(type(devign_dataset))
    # print(type(sard_dataset))
    # return

    df1_code = devign_dataset[['code']].copy()
    df2_code = sard_dataset[['code']].copy()

    # Concatenate the DataFrames
    dataset = pd.concat([df1_code, df2_code], ignore_index=True)
    
    # Initialize the tokenizer and vocabulary manager
    normalizer = CXXNormalization()
    tokenizer = CXXNodeTokenizer()
    vocab_mgr = CXXVocabularyManager()

    # Normalize the code
    dataset['code'] = dataset['code'].apply(lambda x: normalizer.normalization(x))

    # Tokenize the code and update the vocabulary
    tokens_lists = dataset['code'].apply(lambda x: tokenizer.tokenize_source(x)).tolist()
    vocab_mgr.update_vocab(tokens_lists)
    
    # Save the vocabulary to files
    vocab_mgr.save_vocab_files()
    logger.info("Vocabulary saved to vocab.json")

def train(dataset=DATASET_PATH[0], model="lstm"):
    # Read vocabulary from file
    vocab = {}
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Load the training dataset
    train = pd.read_pickle(dataset)

    # Get only 10% sample
    # Define the target column name (adjust if different)
    target_column = 'target'

    # Extract a 10% balanced subset
    subset_size = 0.01  # 10% of the dataset
    train = train.groupby(target_column).sample(frac=subset_size, random_state=None)

    # Normalize source code
    normalizer = CXXNormalization()
    train['code'] = train['code'].apply(lambda x: normalizer.normalization(x))

    # Tokenize
    tokenizer = CXXNodeTokenizer()
    train['tokens'] = train['code'].apply(lambda x: tokenizer.tokenize_source(x))

    # Count Length Token
    # Calculate token lengths (number of tokens per row)
    token_lengths = train['tokens'].apply(len)

    # Compute statistics
    min_length = token_lengths.min()
    max_length = token_lengths.max()
    median_length = token_lengths.median()
    average_length = token_lengths.mean()

    # Print the results
    print("Token Length Statistics:")
    print(f"Minimum length: {min_length}")
    print(f"Maximum length: {max_length}")
    print(f"Median length: {median_length:.2f}")
    print(f"Average length: {average_length:.2f}")

    # Convert the code to AST
    train['code'] = train['code'].apply(lambda x: parse_ast(x))

    # Tokenize the code
    tokenizer = CXXNodeTokenizer()
    train['tokens'] = train['code'].apply(lambda x: tokenizer.tokenize(x))
    logger.info(f"Number of training samples: {len(train)}")

    # Convert tokens to indices
    labels = train['target']
    logger.info("Converting tokens to IDs...")
    train['tokens'] = convert_tokens_to_ids(train['tokens'], vocab)
    maxlen = 512

    # Prepare data loaders
    train_loader, test_loader = prepare_data(
        train['tokens'], labels, vocab, max_len=maxlen, batch_size=128)

    # Train models
    embed_size = 128    
    heads = 8
    num_layers = 2
    # model = TransformerModel(vocab_size=vocab_size, embed_size=128, heads=8, num_layers=num_layers)
    # train_model(model, train_loader, epochs=2)

    # LSTM
    if model == "lstm":
        embedding_dim = 256
        hidden_dim = 256
        num_layers = 2
        num_classes = 2  # Binary classification
        dropout = 0.2
        batch_size = 32
        num_epochs = 10
        learning_rate = 0.001

        # Initialize model, loss function, and optimizer
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Train the model
        train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s = train_lstm(model, train_loader, test_loader, criterion, optimizer, num_epochs, device, num_classes)

        # Plot the results
        lstm_plot_metrics(train_losses, test_losses, test_accuracies, test_precisions, test_recalls, test_f1s)

        # print("Example first tokens:", train['tokens'].iloc[0])  # Debugging output
        # train['tokens'] = train['tokens'].apply(lambda x: [vocab.get(token, 0) for token in x])  # Default to 0 for unknown tokens
        # logger.info("Tokenization completed.")

        # print("Example tokens:", train['tokens'].iloc[0])  # Debugging output

        # # Covert tokens to indices
        # train['token_indices'] = train['tokens'].apply(lambda x: [vocab.get(token, 0) for token in x])  # Default to 0 for unknown tokens

        # # Train the model
        # model = LSTMModel(vocab_size, embedding_dim, hidden_dim=256, num_layers=2, num_classes=2)
        # logger.info("LSTM model initialized.")




def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess~.')
    parser.add_argument('-i', '--input', default='mutrvd', choices=['mutrvd', 'TrVD', 'Devign'],
                        help='training dataset type', type=str, required=False)

    # Optional
    parser.add_argument('-g', '--gen-vocab', action='store_true',
                        help='Generate vocabulary file', required=False)
    parser.add_argument('-t', '--train', action='store_true',
                        help='Train the model', required=False)
    args = parser.parse_args()
    return args

def parse_ast(source):
    CPP_LANGUAGE = Language(tree_sitter_cpp.language())  

    parser = Parser()
    # parser.set_language(CPP_LANGUAGE)  # set the parser for certain language
    parser.language = CPP_LANGUAGE
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    return tree

args = parse_options()

class VulFuncPipeline:
    def __init__(self):
        pass

    # run for processing raw to train
    def run(self, args):
        # dataset = args.input
        # train = pd.read_pickle('datasets/'+dataset+'/train.pkl')[1:2]
        # print(train['code'].tolist())
        # train['code'] = train['code'].apply(parse_ast)

        # Build vocabs.json
        if args.gen_vocab:
            logger.info("[+] Generating vocabulary...")
            generate_vocab()
        if args.train:
            logger.info("[+] Training the model...")
            train()

if __name__ == '__main__':
    ppl = VulFuncPipeline()
    ppl.run(args)

