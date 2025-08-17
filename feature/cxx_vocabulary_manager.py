import os
import json

# Configuration
DATA_DIR = "CXX_AST_DATA"
VOCAB_DIR = os.path.join(DATA_DIR, "vocab")

class CXXVocabularyManager:
    """
    Maintains and updates a fixed-size vocabulary with wildcard mappings for C++ tokens.
    """

    def __init__(self, max_vocab_size=30000):
        self.max_vocab_size = max_vocab_size
        self.tokens = {}
        self.next_index = 0
        self.combined_tokens = set()

    def update_vocab(self, token_lists):
        for tokens in token_lists:
            for token in tokens:
                if token not in self.tokens:
                    if self.next_index < self.max_vocab_size:
                        if isinstance(token, bytes):
                            token = token.decode('utf-8')
                        # wildcard = f"token_{self.next_index:05d}"
                        self.tokens[token] = self.next_index
                        self.next_index += 1

        # tokens_str_keys = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in self.tokens.items()}


    # def replace_tokens_with_wildcards(self, token_list):
    #     return [self.tokens.get(token, "<UNK>") for token in token_list]

    def save_vocab(self, filepath="vocab.json"):
        sorted_items = sorted(self.tokens.items(), key=lambda x: x[0])
        self.tokens = {k: idx for idx, (k, _) in enumerate(sorted_items)}

        # Save the vocabulary to a JSON file (from dict to JSON)
        with open(filepath, "w") as f:
            json.dump(self.tokens, f, indent=4)

    def save_vocab_files(self):
        """Save vocabulary files to disk."""
        print("[+] Saving vocabulary files...")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VOCAB_DIR, exist_ok=True)

        with open(os.path.join(VOCAB_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
            self.combined_tokens = sorted(self.tokens.keys())
            # self.combined_tokens = sorted(
            #     self.normal_tokens.union(self.anomalous_tokens)
            # )
            # Add <unk> and <pad> at the end of the combined tokens
            self.combined_tokens.append("<unk>")
            self.combined_tokens.append("<pad>")

            for token in self.combined_tokens:
                f.write(f"{token}\n")

        vocab_index = {token: idx for idx,
                       token in enumerate(self.combined_tokens)}

        with open(os.path.join(VOCAB_DIR, "vocab.json"), "w", encoding="utf-8") as f_json:
            import json
            json.dump(vocab_index, f_json, ensure_ascii=False, indent=4)

        print(f"Vocabulary files saved to {VOCAB_DIR}")
        # print(f"  - Normal tokens: {len(self.normal_tokens)}")
        # print(f"  - Anomalous tokens: {len(self.anomalous_tokens)}")
        print(f"  - Combined total: {len(self.combined_tokens)}")

# Example usage
if __name__ == "__main__":
    from cxx_node_tokenizer import CXXNodeTokenizer
    cxx_code_1 = """
        int FUN1() { int VAR2 = 1;    int VAR3 = 2;    return VAR2 + VAR3; }
    """

    cxx_code_2 = """
        void FUN2() { int VAR4 = 3;    int VAR5 = 4;    return VAR4 * VAR5; }
    """

    tokenizer = CXXNodeTokenizer()
    tokens = [tokenizer.tokenize(cxx_code_1), tokenizer.tokenize(cxx_code_2)]

    vocab_mgr = CXXVocabularyManager()
    vocab_mgr.update_vocab(tokens)
    print(vocab_mgr.tokens)
    vocab_mgr.save_vocab_files()