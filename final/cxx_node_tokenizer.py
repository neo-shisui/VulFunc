import re
import keyword
import tree_sitter_cpp
from tree_sitter import Language, Parser

from cxx_ast_traversal import get_root_paths, get_sequences

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
        if source is None:
            return []

        # Parse the source code into an AST
        tree = self.parser.parse(source.encode('utf-8').decode('unicode_escape').encode())

        # Convert the AST to a sequence of tokens
        sequences = []
        get_sequences(tree, sequences)
        # print(source)
        # print(len(sequences))

        # Preserve unique tokens
        # self.token_sequences = list(set(self.token_sequences))
        return sequences
    
    def tokenize(self, ast):
        # Convert the AST to a sequence of tokens
        get_sequences(ast, self.token_sequences)
        return self.token_sequences
       
    def get_token_sequences(self):
        return self.token_sequences

# Example usage
if __name__ == "__main__":
    cxx_code = """
        void CWE15_External_Control_of_System_or_Configuration_Setting__w32_01_bad()
{
    char * data;
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
    /* POTENTIAL FLAW: set the hostname to data obtained from a potentially external source */
    if (!SetComputerNameA(data))
    {
        printLine("Failure setting computer name");
        exit(1);
    }
}
    """

    print(cxx_code)

    tokenizer = CXXNodeTokenizer()
    tokens = tokenizer.tokenize_source(cxx_code)
    print("Tokens:", tokens)
    print("Number of unique tokens:", len(tokens))