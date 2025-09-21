# from tree_sitter import Parser
import tree_sitter
import tree_sitter_cpp
# from tree_sitter_languages import get_language

CPP_LANGUAGE = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser()
parser.language = CPP_LANGUAGE

# # Load C language
# C_LANGUAGE = get_language("c")

# parser = Parser()
# parser.set_language(C_LANGUAGE)

code = b"""
int main() {
    int x = 0;
    if (x < 10) {
        for (int i=0; i<10; i++) {
            x += i;
        }
    }
    return x;
}
"""

tree = parser.parse(code)
root = tree.root_node

def extract_blocks(node, depth=0):
    """Recursively extract blocks (L1, L2, L3...)"""
    blocks = []
    if node.type == "compound_statement":  # { ... }
        block_text = code[node.start_byte+1:node.end_byte-1].decode("utf-8").strip()
        blocks.append((depth, block_text))

        depth = depth + 1

    for child in node.children:
        blocks.extend(extract_blocks(child, depth))
    return blocks


# Walk AST and collect blocks
all_blocks = []
for child in root.children:
    if child.type == "function_definition":
        all_blocks.extend(extract_blocks(child, depth=1))

# Print result
for depth, block in all_blocks:
    print(f"L{depth}:\n{block}\n")
