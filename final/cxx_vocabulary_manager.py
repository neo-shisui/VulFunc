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
        # self.tokens = {}
        self.tokens = {}
        self.count = 0

    def update_vocab(self, token_lists):
        for tokens in token_lists:
            if self.count >= self.max_vocab_size:
                break
            for token in tokens:
                if isinstance(token, bytes):
                    token = token.decode('utf-8')
                if token in self.tokens:
                    self.tokens[token] += 1
                elif self.count < self.max_vocab_size:
                    self.tokens[token] = 1
                    self.count += 1 

        # tokens_str_keys = {k.decode('utf-8') if isinstance(k, bytes) else k: v for k, v in self.tokens.items()}


    # def replace_tokens_with_wildcards(self, token_list):
    #     return [self.tokens.get(token, "<UNK>") for token in token_list]

    # def save_vocab(self, filepath="vocab.json"):
    #     sorted_items = sorted(self.tokens.items(), key=lambda x: x[0])
    #     self.tokens = {k: idx for idx, (k, _) in enumerate(sorted_items)}

    #     # Save the vocabulary to a JSON file (from dict to JSON)
    #     with open(filepath, "w") as f:
    #         json.dump(self.tokens, f, indent=4)

    def save_vocab(self, index_filepath="vocab.json", freq_filepath="frequencies.json"):
        """
        Saves two files: one mapping tokens to indices and another with token frequencies.
        Args:
            index_filepath: Path to save the token-to-index JSON file.
            freq_filepath: Path to save the token-to-frequency JSON file.
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VOCAB_DIR, exist_ok=True)

        # Create token-to-index mapping (sorted alphabetically by token)
        sorted_items = sorted(self.tokens.items(), key=lambda x: x[0])
        token_to_index = {k: idx for idx, (k, _) in enumerate(sorted_items)}

        # Save token-to-index mapping
        with open(os.path.join(VOCAB_DIR, index_filepath), "w", encoding='utf-8') as f:
            json.dump(token_to_index, f, indent=4, ensure_ascii=False)

        # Save token-to-frequency mapping (sorted by frequency, descending)
        sorted_by_freq = sorted(self.tokens.items(), key=lambda x: x[1], reverse=True)
        token_to_freq = {k: v for k, v in sorted_by_freq}

        with open(os.path.join(VOCAB_DIR, freq_filepath), "w", encoding='utf-8') as f:
            json.dump(token_to_freq, f, indent=4, ensure_ascii=False)

        print(f"[+] Vocabulary size: {len(self.tokens)}")

    def save_vocab_files(self):
        """Save vocabulary files to disk."""
        print("[+] Saving vocabulary files...")
        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(VOCAB_DIR, exist_ok=True)

        with open(os.path.join(VOCAB_DIR, "vocab.txt"), "w", encoding="utf-8") as f:
            self.tokens = sorted(self.tokens)
            # self.combined_tokens = sorted(
            #     self.normal_tokens.union(self.anomalous_tokens)
            # )
            # Add <unk> and <pad> at the end of the combined tokens
            self.tokens.append("<unk>")
            self.tokens.append("<pad>")

            for token in self.tokens:
                f.write(f"{token}\n")

        vocab_index = {token: idx for idx,
                       token in enumerate(self.tokens)}

        with open(os.path.join(VOCAB_DIR, "vocab.json"), "w", encoding="utf-8") as f_json:
            import json
            json.dump(vocab_index, f_json, ensure_ascii=False, indent=4)

        print(f"Vocabulary files saved to {VOCAB_DIR}")
        # print(f"  - Normal tokens: {len(self.normal_tokens)}")
        # print(f"  - Anomalous tokens: {len(self.anomalous_tokens)}")
        print(f"  - Combined total: {len(self.tokens)}")

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
    tokens = [tokenizer.tokenize_source(cxx_code_1), tokenizer.tokenize_source(cxx_code_2)]

    vocab_mgr = CXXVocabularyManager()
    vocab_mgr.update_vocab(tokens)
    print(vocab_mgr.tokens)
    vocab_mgr.save_vocab_files()