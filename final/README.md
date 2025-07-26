# VulFunc

## C/C++ Source Code Vunerability Detection Pipeline

### 1. VulnerabilityDictionaryCXX

### 2. CXX_Normalization

**Purpose:** Normalize source code C++ by removing comments and renaming user-defined variables/functions.

**Key Methods:**

* `normalization()`

---

### 3. CXX_AST_Parser

**Purpose:** Generate Abstract Syntax Trees (AST) from C/C++ source files using tree-sitter.

**Key Methods:**

* Support function:
    * `get_max_depth()`
    * `get_tree_size()`
    * `needs_splitting()`

* Class ASTNode: 
    * `is_leaf_node()`: Whether the node has children
    * `get_token()`: Label/type of the node in AST
    * `get_children()`: List of child ASTNode objects (or empty if leaf)

---

### 4. CXX_AST_Traversal
**Purpose:** Generate Abstract Syntax Trees (AST) from C/C++ source files using tree-sitter.

**Key Methods:**

* `get_sequences()`: Traverse AST and append token sequence to a list
* `get_root_paths()`: Build all paths from root to each leaf in the AST
* `get_blocks()`: Splits AST into manageable control blocks

### 5. CXX_Node_Tokenizer

**Purpose:** Tokenize AST nodes (sub-tree); normalize identifiers for ML modeling.

**Key Methods:**

* `tokenize()`

---

### 6. CXX_Vocabulary_Manager

**Purpose:** Maintain a consistent vocabulary for C/C++ node tokens.

**Key Methods:**

* `update_vocab(token_lists)`
* `save_vocab()`

---

### 7. CXX_Token_Embedding

**Purpose:** 

**Key Methods:**

---

### 8. LSTM_Model

## Usage

```
$ pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

```
$ export PYTHONPATH=$PWD
```

## References
* [TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection](https://github.com/XUPT-SSS/TrVD/tree/main)