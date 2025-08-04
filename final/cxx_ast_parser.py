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

# Use for construct subTree
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

def get_float_bytes(is_float):
    return 4 if is_float else 8  # float: 4 bytes, double: 8 bytes

def get_int_bytes(x):
    x = abs(x)
    if x <= 127:
        return "1byte_number"  # int8
    elif x <= 32767:
        return "2byte_number"  # int16
    elif x <= 2147483647:
        return "4byte_number"  # int32
    return "8byte_number"  # int64

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
                if self.node.type == "number_literal":
                    # if float
                    if '.' in token:
                        token = "<float>"
                    else:
                        if token.isdigit():
                            token = get_int_bytes(int(token))
                        elif '0x' in token or '0X' in token:
                            try:
                                token = get_int_bytes(int(token, 16))
                            except ValueError:
                                token = "<num>"
                        else:
                            token = "<num>"
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

def print_ast(node, level=0, text_tree=''):
    if not node:
        return
    if isinstance(node, list):
        for n in node:
            print_ast(n, level, text_tree)
        return
    if not isinstance(node, tree_sitter.Tree):
        children = node.children
        name = node.type
        token = ''
        if is_leaf_node(node):
            token = node.text
            if type(token) is bytes:
                token = token.decode('utf-8')
    else:
        children = node.root_node.children
        name = node.root_node.type
        token = ''
        if is_leaf_node(node.root_node):
            token = node.root_node.text.decode('utf-8')
            # if node.root_node.type == "number_literal":
            #     token = "<num

    # if len(children) == 0:
    #     return

    print(' ' * level + name + ' ' + token)
    with open('ast.txt', 'a') as f:
        f.write(' ' * level + name + ' ' + token + '\n')
    text_tree = text_tree + ' ' * level + name + '\n'
    for child in children:
        print_ast(child, level + 1, text_tree)

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

    source = """void* bad(int x){
    char * data;
    vector<char *> dataVector;
    char dataBuffer[100] = "";
    data = dataBuffer;
    {
        WSADATA wsaData;
        BOOL wsaDataInit = FALSE;
        SOCKET listenSocket = INVALID_SOCKET;
        SOCKET acceptSocket = INVALID_SOCKET;
        struct sockaddr_in service;
        int recvResult;
        do
        {
            if (WSAStartup(MAKEWORD(2,2), &wsaData) != NO_ERROR)
            {
                break;
            }
            wsaDataInit = 1;
            listenSocket = socket(PF_INET, SOCK_STREAM, 0);
            if (listenSocket == INVALID_SOCKET)
            {
                break;
            }
            memset(&service, 0, sizeof(service));
            service.sin_family = AF_INET;
            service.sin_addr.s_addr = INADDR_ANY;
            service.sin_port = htons(LISTEN_PORT);
            if (SOCKET_ERROR == bind(listenSocket, (struct sockaddr*)&service, sizeof(service)))
            {
                break;
            }
            if (SOCKET_ERROR == listen(listenSocket, LISTEN_BACKLOG))
            {
                break;
            }
            acceptSocket = accept(listenSocket, NULL, NULL);
            if (acceptSocket == INVALID_SOCKET)
            {
                break;
            }
            /* INCIDENTAL CWE 188 - reliance on data memory layout
             * recv and friends return "number of bytes" received
             * char's on our system, however, may not be "octets" (8-bit
             * bytes) but could be just about anything.  Also,
             * even if the external environment is ASCII or UTF8,
             * the ANSI/ISO C standard does not dictate that the
             * character set used by the actual language or character
             * constants matches.
             *
             * In practice none of these are usually issues...
             */
            /* FLAW: read the new hostname from a network socket */
            recvResult = recv(acceptSocket, data, 100 - 1, 0);
            if (recvResult == SOCKET_ERROR || recvResult == 0)
            {
                break;
            }
            data[recvResult] = '\0';
        }
        while (0);
        if (acceptSocket != INVALID_SOCKET)
        {
            closesocket(acceptSocket);
        }
        if (listenSocket != INVALID_SOCKET)
        {
            closesocket(listenSocket);
        }
        if (wsaDataInit)
        {
            WSACleanup();
        }
    }
"""

    source = """
void bad(int x){
    int a = 10.1;
}
"""

    # print("Source code:", source)
    # Parse the source code into an AST
    tree = parser.parse(source.encode('utf-8').decode('unicode_escape').encode())
    text = ''
    print_ast(tree, 0, text)
    # with open('ast.txt', 'w') as f:
    #     f.write(text)

    # sequence = []
    # get_sequences(tree, sequence)
    # print(sequence)
    # for i, seq in enumerate(sequence):
    #     print(f"Token {i}: {seq}")
    # print("Token sequence:", sequence)

    # x = ast_to_json(tree)

    # print("AST in JSON format:", json.dumps(x, indent=2))