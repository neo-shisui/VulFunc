# VulFunc

## C/C++ Source Code Vunerability Detection Pipeline

### 1. VulnerabilityDictionaryCXX

---

### 2. CXX_AST_Parser

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

### 3. CXX_AST_Traversal
**Purpose:** Generate Abstract Syntax Trees (AST) from C/C++ source files using tree-sitter.

**Key Methods:**

* `get_sequences()`: Traverse AST and append token sequence to a list
* `get_root_paths()`: Build all paths from root to each leaf in the AST
* `get_blocks()`: Splits AST into manageable control blocks

## References
* [TrVD: Deep Semantic Extraction via AST Decomposition for Vulnerability Detection](https://github.com/XUPT-SSS/TrVD/tree/main)