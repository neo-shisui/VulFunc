import json
import tree_sitter

def get_max_depth(node):
    if not node:
        return 0
    # leaf node touched
    if len(node.children) == 0:
        return 1

    children = node.children
    max_depth = -1
    for child in children:
        depth = get_max_depth(child)
        max_depth = depth if (depth > max_depth) else max_depth
    return max_depth + 1

def get_tree_size(node):

    if not node:
        return 0
    # leaf node touched
    if len(node.children) == 0:
        return 1
    tr_size = 1
    children = node.children
    for child in children:
        node_num = get_tree_size(child)
        tr_size += node_num
    return tr_size

def needs_splitting(node, max_depth=8, max_size=40):
    '''
    split if the depth or size of the sub-tree exceeds certain thresholds
    :param node:
    :return:
    '''
    tr_depth = get_max_depth(node)
    tr_size = get_tree_size(node)
    if tr_depth > max_depth or tr_size > max_size:
        return True
    return False

class ASTNode(object):
    def __init__(self, node, do_split=True):
        self.node = node
        self.do_split = do_split
        self.is_leaf = self.is_leaf_node()
        self.token = self.get_token()
        self.children = self.add_children()

    # Checks if the node is a leaf (no children)
    # - If self.node is a Node, checks self.node.children.
    # - If self.node is a Tree, checks the root node’s children (self.node.root_node.children)
    def is_leaf_node(self):
        if not isinstance(self.node, tree_sitter.Tree):
            return len(self.node.children) == 0
        else:
            return len(self.node.root_node.children) == 0

    # Retrieves the node’s token:
    # - For non-leaf nodes: Returns the node’s type (e.g., 'function_definition', 'if_statement').
    # - For leaf nodes: Returns the actual source code text (e.g., a variable name or keyword).
    def get_token(self, lower=True):
        if not isinstance(self.node, tree_sitter.Tree):
            token = self.node.type
            if self.is_leaf:
                token = self.node.text
                if self.node.type == "number_literal":
                    token = "<num>"
            return token
        else:
            token = self.node.root_node.type
            if self.is_leaf:
                token = self.node.root_node.text
                if self.node.root_node.type == "number_literal":
                    token = "<num>"
            return token

    # Populates the self.children list based on the node’s children and the do_split flag.
    # - If the node is a leaf: Returns an empty list ([]).
    # - If do_split is False: Creates an ASTNode for each child and includes all children.
    # - If do_split is True and the node’s token is in a specific list (e.g., 'function_definition', 'if_statement'):
    def add_children(self):
        # from prepare_data import needsSplitting
        if self.is_leaf:
            return []
        children = self.node.children
        if not self.do_split:
            return [ASTNode(child, self.do_split) for child in children]
        else:
            if self.token in ['function_definition', 'if_statement', 'try_statement', 'for_statement',
                              'switch_statement',
                              'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
                # find first compound_statement
                body_idx = 0
                for child in children:
                    if child.type == 'compound_statement' or child.type == 'expression_statement':
                        break
                    body_idx += 1
                return [ASTNode(children[c], self.do_split) for c in range(0, body_idx)]
            else:
                return [ASTNode(child, self.do_split) for child in children]

    def get_children(self):
        return self.children

class SingleNode(ASTNode):
    def __init__(self, node):
        self.node = node
        self.is_leaf = self.is_leaf_node()
        self.token = self.get_token()
        self.children = []

    def is_leaf_node(self):
        if not isinstance(self.node, tree_sitter.Tree):
            return len(self.node.children) == 0
        else:
            return len(self.node.root_node.children) == 0

    def get_token(self, lower=True):
        if not isinstance(self.node, tree_sitter.Tree):
            token = self.node.type
            if self.is_leaf:
                token = self.node.text.decode('utf-8')
                # if self.node.type == "number_literal":
                #     token = "<num>" 
            return token
        else:
            token = self.node.root_node.type
            if self.is_leaf:
                token = self.node.root_node.text.decode('utf-8')
                # if self.node.root_node.type == "number_literal":
                #     token = "<num>" 
            return token

def is_leaf_node(node):
    if not isinstance(node, tree_sitter.Tree):
        return len(node.children) == 0
    else:
        return len(node.root_node.children) == 0

def print_ast(node, level=0):
    if not node:
        return
    if isinstance(node, list):
        for n in node:
            print_ast(n, level)
        return
    if not isinstance(node, tree_sitter.Tree):
        children = node.children
        name = node.type
        token = node.text
    else:
        children = node.root_node.children
        name = node.root_node.type
        token = node.root_node.text

    # if len(children) == 0:
    #     return

    print(' ' * level + name)
    for child in children:
        print_ast(child, level + 1)
        pass

def get_sequences(node, sequence: list):
    current = SingleNode(node)

    if not isinstance(node, tree_sitter.Tree):
        name = node.type
    else:
        name = node.root_node.type

    if name == 'comment':
        return
    else:
        sequence.append(current.get_token())

    if not isinstance(node, tree_sitter.Tree):
        for child in node.children:
            get_sequences(child, sequence)
    else:
        for child in node.root_node.children:
            get_sequences(child, sequence)
    if current.get_token().lower() == 'compound_statement':
        sequence.append('End')

if __name__ == '__main__':
    import tree_sitter_cpp
    CPP_LANGUAGE = tree_sitter.Language(tree_sitter_cpp.language())
    parser = tree_sitter.Parser()
    parser.language = CPP_LANGUAGE

    source = """
static void FUN_1(DeviceState *VAR_1, Error **VAR_2)
{
     VirtIODevice *VAR_3 = FUN_2(VAR_1);
     V9fsVirtioState *VAR_4 = FUN_3(VAR_1);
     V9fsState *VAR_5 = &VAR_4->VAR_6;
     FUN_4(VAR_3);
     FUN_5(VAR_5, VAR_2);
}"""
#     VirtIODevice *vdev = VIRTIO_DEVICE(dev);

#     V9fsVirtioState *v = VIRTIO_9P(dev);

#     V9fsState *s = &v->state;

#     virtio_cleanup(vdev);

#     v9fs_device_unrealize_common(s, errp);
# }"""
    print("Source code:", source)
    # Parse the source code into an AST
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    print_ast(tree)

    sequence = []
    get_sequences(tree, sequence)
    print(sequence)
    # for i, seq in enumerate(sequence):
    #     print(f"Token {i}: {seq}")
    # print("Token sequence:", sequence)

    # x = ast_to_json(tree)

    # print("AST in JSON format:", json.dumps(x, indent=2))