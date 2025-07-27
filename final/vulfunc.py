import argparse
import copy
import pandas as pd
import os
import re
import json
import pickle
import keyword
import logging
import tree_sitter_cpp
from tree_sitter import Language, Parser


import sys
from cxx_node_tokenizer import CXXNodeTokenizer
from cxx_vocabulary_manager import CXXVocabularyManager
from cxx_normalization import CXXNormalization
from cxx_token_embedding import CXXTokenEmbedding
from lstm import LSTMModel
from cxx_ast_traversal import get_root_paths, get_sequences


# Configurations
DATASET_PATH = [
    '../datasets/Devign/train.pkl',
]

VOCAB_PATH = 'CXX_AST_DATA/vocab/vocab.json'

logger = logging.getLogger(__name__)
logging.basicConfig(filename='vulfunc.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_vocab():
    """
    Generate vocabulary from the training dataset.
    """
    # Load the training dataset
    train = pd.read_pickle(DATASET_PATH[0])[:1000]
    print(len(train))
    # return
    # print(train['code'].tolist())
    
    # Initialize the tokenizer and vocabulary manager
    normalizer = CXXNormalization()
    tokenizer = CXXNodeTokenizer()
    vocab_mgr = CXXVocabularyManager()

    # Normalize the code
    train['code'] = train['code'].apply(lambda x: normalizer.normalization(x))

    # Tokenize the code and update the vocabulary
    tokens_lists = train['code'].apply(lambda x: tokenizer.tokenize_source(x)).tolist()
    vocab_mgr.update_vocab(tokens_lists)
    
    # Save the vocabulary to files
    vocab_mgr.save_vocab_files()
    print("Vocabulary saved to vocab.json")

def train():
    # Read vocabulary from file
    vocab = {}
    with open(VOCAB_PATH, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    logger.info(f"Vocabulary size: {vocab_size}")
    
    # Load the training dataset
    train = pd.read_pickle(DATASET_PATH[0])[:80]

    # Convert the code to AST
    # Print code with target is 0
    for i in range(0, len(train)):
        if train['target'].iloc[i] == 0:
            print(train['code'].iloc[i])
    # print(train['code'].iloc[1])  # Debugging output
    # print(train['target'].iloc[1])  # Debugging output
    train['code'] = train['code'].apply(lambda x: parse_ast(x))

    # Tokenize the code
    tokenizer = CXXNodeTokenizer()
    train['tokens'] = train['code'].apply(lambda x: tokenizer.tokenize(x))
    logger.info(f"Number of training samples: {len(train)}")

    # Convert tokens to indices
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


