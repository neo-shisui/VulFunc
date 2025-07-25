import copy
import tree_sitter

try:
    from cxx_ast_parser import ASTNode, SingleNode, needs_splitting, get_tree_size, get_max_depth
except:
    from feature.cxx_ast_parser import ASTNode, SingleNode, needs_splitting, get_tree_size, get_max_depth

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

def get_root_paths(node, sequences: list, cur_path: list):
    '''
    collect all paths that originate from the root to each leaf node
    :param node:
    :param sequences:
    :param cur_path:
    :return:
    '''
    current = SingleNode(node)

    if current.is_leaf_node():

        if not isinstance(node, tree_sitter.Tree):
            name = node.type
        else:
            name = node.root_node.type

        if name == 'comment':
            return
        else:
            root_path = copy.deepcopy(cur_path)
            cur_token = current.get_token()
            root_path.append(cur_token)
            sequences.append(root_path)
            return
    else:

        if not isinstance(node, tree_sitter.Tree):
            name = node.type
        else:
            name = node.root_node.type

        if name == 'comment':
            return
        else:
            cur_path.append(current.get_token())
            if not isinstance(node, tree_sitter.Tree):
                for child in node.children:
                    par_path = copy.deepcopy(cur_path)
                    get_root_paths(child, sequences, par_path)
            else:
                # for _, child in node.root_node.children():
                for child in node.root_node.children:
                    par_path = copy.deepcopy(cur_path)
                    get_root_paths(child, sequences, par_path)

def get_blocks(node, block_seq):
    if isinstance(node, list):
        return
    elif not isinstance(node, tree_sitter.Tree):
        children = node.children
        name = node.type
    else:
        children = node.root_node.children
        name = node.root_node.type

    if name == 'comment':
        return
    print(name)
    if name in ['function_definition', 'if_statement', 'try_statement', 'for_statement', 'switch_statement',
                'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
        # split further?
        do_split = needs_splitting(node)
        # print(do_split, 'tr_size ', get_tree_size(node), ' tr_depth ', get_max_depth(node))

        if not do_split:
            block_seq.append(ASTNode(node, False))
            return
        else:
            block_seq.append(ASTNode(node, True))
            # find first compound_statement
            body_idx = 0
            for child in children:
                if child.type == 'compound_statement': # or child.type == 'expression_statement':
                    break
                body_idx += 1

            skip = body_idx

            for i in range(skip, len(children)):
                child = children[i]
                if child.type == 'comment':
                    continue
                if child.type not in ['function_definition', 'if_statement', 'try_statement', 'for_statement',
                                      'switch_statement',
                                      'while_statement', 'do_statement']:
                    block_seq.append(ASTNode(child, needs_splitting(child)))
                get_blocks(child, block_seq)
    elif name == 'compound_statement':
        # block_seq.append(ASTNode(name))
        do_split = needs_splitting(node)

        if not isinstance(node, tree_sitter.Tree):
            for child in node.children:
                if child.type == 'comment':
                    continue

                if child.type not in ['if_statement', 'try_statement', 'for_statement', 'switch_statement',
                                      'while_statement', 'do_statement', 'catch_clause']:
                    block_seq.append(ASTNode(child, needs_splitting(child)))
                else:
                    get_blocks(child, block_seq)
        else:
            for child in node.root_node.children:
                if child.type == 'comment':
                    continue
                if child.type not in ['if_statement', 'try_statement', 'for_statement', 'switch_statement',
                                      'while_statement', 'do_statement', 'catch_clause', 'case_statement']:
                    block_seq.append(ASTNode(child, needs_splitting(child)))
                get_blocks(child, block_seq)
        # block_seq.append(ASTNode('End'))
    else:
        if isinstance(node, list):
            return
        elif not isinstance(node, tree_sitter.Tree):
            for child in node.children:
                get_blocks(child, block_seq)
        else:
            for child in node.root_node.children:
                get_blocks(child, block_seq)

if __name__ == '__main__':
    import tree_sitter_cpp
    from tree_sitter import Language, Parser

    parser = Parser()
    CPP_LANGUAGE = Language(tree_sitter_cpp.language())  
    parser.language = CPP_LANGUAGE

    code = """
static void FUN1(VAR1 *VAR2)  {  
#if FUN2(VAR3)  
    FUN3(VAR2, VAR4);  
#else  
    if (FUN4(VAR2->VAR5)) {  
        FUN3(VAR2, VAR4);  
        return;  
    }  
    FUN5(VAR6, VAR7[FUN6(VAR2->VAR8)],  VAR7[FUN7(VAR2->VAR8)]);  
#endif  
}
"""

    tree = parser.parse(bytes(code, 'utf8'))
    root_node = tree.root_node

    sequences = []
    get_sequences(root_node, sequences)
    print(sequences)

    # root_paths = []
    # get_root_paths(root_node, root_paths, [])
    # for path in root_paths:
    #     print(path) 
    # print('Total paths:', len(root_paths))
    # print(root_paths)

    # block_seq = []
    # get_blocks(root_node, block_seq)
    # for block in block_seq:
    #     print(block.get_token())
    #     sequences = []
    #     get_sequences(block.node, sequences)
    #     print(sequences)