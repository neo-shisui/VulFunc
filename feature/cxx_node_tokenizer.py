import re
import keyword

class CXXNodeTokenizer:
    """
    Tokenizes C++ AST Nodes and normalizes identifiers.
    """

    def __init__(self):
        self.token_sequences = []
        self.identifier_map = {}
        self.var_counter = 0
        self.func_counter = 0

    def tokenize(self, java_code):
        # Basic tokenizer based on word and operator boundaries
        tokens = re.findall(r"\b\w+\b|[^\s\w]", java_code)
        self.token_sequences.append(tokens)
        return tokens

    def normalize_identifiers(self, tokens):
        normalized = []
        for tok in tokens:
            if keyword.iskeyword(tok) or tok in ["int", "String", "public", "class", "void"]:
                normalized.append(tok)
            elif re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", tok):
                if tok not in self.identifier_map:
                    if tok[0].isupper():  # likely a class or type
                        self.identifier_map[tok] = f"Type_{len(self.identifier_map)}"
                    elif tok.endswith("()"):  # likely a method
                        self.identifier_map[tok] = f"func_{self.func_counter}"
                        self.func_counter += 1
                    else:
                        self.identifier_map[tok] = f"var_{self.var_counter}"
                        self.var_counter += 1
                normalized.append(self.identifier_map[tok])
            else:
                normalized.append(tok)
        return normalized

    def get_token_sequences(self):
        return self.token_sequences

# Example usage
if __name__ == "__main__":
    java_code = """
    public class HelloWorld {
        public static void main(String[] args) {
            String message = "Hello, World!";
            System.out.println(message);
        }
    }
    """

    tokenizer = CXXNodeTokenizer()
    tokens = tokenizer.tokenize(java_code)
    normalized = tokenizer.normalize_identifiers(tokens)
    print("Original Tokens:", tokens)
    print("Normalized Tokens:", normalized)