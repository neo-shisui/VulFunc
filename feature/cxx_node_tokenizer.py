import re
import keyword
import tree_sitter_cpp
from tree_sitter import Language, Parser
try:
    from cxx_ast_traversal import get_root_paths, get_sequences
except:
    from feature.cxx_ast_traversal import get_root_paths, get_sequences

class CXXNodeTokenizer:
    """
    Tokenizes C++ AST Nodes and normalizes identifiers.
    """

    def __init__(self):
        self.token_sequences = []
        self.identifier_map = {}
        self.var_counter = 0
        self.func_counter = 0

        # AST Parser
        self.parser = Parser()
        self.parser.language = Language(tree_sitter_cpp.language())
    
    def tokenize_source(self, source):
        # Parse the source code into an AST
        tree = self.parser.parse(source.encode('utf-8').decode('unicode_escape').encode())

        # Convert the AST to a sequence of tokens
        get_sequences(tree, self.token_sequences)

        # Preserve unique tokens
        # self.token_sequences = list(set(self.token_sequences))
        return self.token_sequences
    
    def tokenize(self, ast):
        # Convert the AST to a sequence of tokens
        get_sequences(ast, self.token_sequences)
        return self.token_sequences
       
    def get_token_sequences(self):
        return self.token_sequences

# Example usage
if __name__ == "__main__":
    cxx_code = """
        int FUN1() { int VAR2 = 1;    int VAR3 = 2;    return VAR2 + VAR3; }
    """

    print(cxx_code)

    tokenizer = CXXNodeTokenizer()
    tokens = tokenizer.tokenize(cxx_code)
    print("Tokens:", tokens)
    print("Number of unique tokens:", len(tokens))