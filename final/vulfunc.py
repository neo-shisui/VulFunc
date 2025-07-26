import argparse
import copy
import pandas as pd
import os
import tree_sitter_cpp
from tree_sitter import Language, Parser


import sys
from cxx_node_tokenizer import CXXNodeTokenizer
from cxx_vocabulary_manager import CXXVocabularyManager
from cxx_normalization import CXXNormalization
# from preprocess.cxx_normalization import CXXNormalization

# Configurations
DATASET_PATH = [
    '../datasets/Devign/train.pkl',
]

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

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess~.')
    parser.add_argument('-i', '--input', default='mutrvd', choices=['mutrvd', 'TrVD', 'Devign'],
                        help='training dataset type', type=str, required=False)

    # Optional
    parser.add_argument('-g', '--gen-vocab', action='store_true',
                        help='Generate vocabulary file', required=False)
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
            print("[+] Generating vocabulary...")
            generate_vocab()

if __name__ == '__main__':
    ppl = VulFuncPipeline()
    ppl.run(args)


