import argparse
import copy
import pandas as pd
import os
import tree_sitter_cpp
from tree_sitter import Language, Parser
from feature.cxx_ast_traversal import get_root_paths, get_sequences

import sys
from feature.cxx_node_tokenizer import CXXNodeTokenizer
from feature.cxx_vocabulary_manager import CXXVocabularyManager
# from preprocess.cxx_normalization import CXXNormalization

def parse_options():
    parser = argparse.ArgumentParser(description='TrVD preprocess~.')
    parser.add_argument('-i', '--input', default='mutrvd', choices=['mutrvd', 'TrVD', 'Devign'],
                        help='training dataset type', type=str, required=False)

    # Optional
    parser.add_argument('-d', '--dump-vocab', action='store_true',
                        help='Dump vocabulary to file', required=False)
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
        dataset = args.input
        train = pd.read_pickle('datasets/'+dataset+'/train.pkl')[1:2]
        print(train['code'].tolist())
        train['code'] = train['code'].apply(parse_ast)

        # Build vocabs.json
        if args.dump_vocab:
            vocab_mgr = CXXVocabularyManager()
            # Assuming you have a method to get token lists, e.g., from the train set

            tokenizer = CXXNodeTokenizer()
            tokens_lists = train['code'].apply(lambda x: tokenizer.tokenize(x)).tolist()

            vocab_mgr.update_vocab(tokens_lists)
            vocab_mgr.save_vocab_files()
            print("Vocabulary saved to vocab.json")

if __name__ == '__main__':
    ppl = VulFuncPipeline()
    ppl.run(args)


