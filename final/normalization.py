from clang.cindex import Index, CursorKind
import re
from pandas import DataFrame

# Keywords up to C11 and C++17; immutable set
keywords = frozenset(['__asm', '__builtin', '__cdecl', '__declspec', '__except', '__export', '__far16', '__far32',
                      '__fastcall', '__finally', '__import', '__inline', '__int16', '__int32', '__int64', '__int8',
                      '__leave', '__optlink', '__packed', '__pascal', '__stdcall', '__system', '__thread', '__try',
                      '__unaligned', '_asm', '_Builtin', '_Cdecl', '_declspec', '_except', '_Export', '_Far16',
                      '_Far32', '_Fastcall', '_finally', '_Import', '_inline', '_int16', '_int32', '_int64',
                      '_int8', '_leave', '_Optlink', '_Packed', '_Pascal', '_stdcall', '_System', '_try', 'alignas',
                      'alignof', 'and', 'and_eq', 'asm', 'auto', 'bitand', 'bitor', 'bool', 'break', 'case',
                      'catch', 'char', 'char16_t', 'char32_t', 'class', 'compl', 'const', 'const_cast', 'constexpr',
                      'continue', 'decltype', 'default', 'delete', 'do', 'double', 'dynamic_cast', 'else', 'enum',
                      'explicit', 'export', 'extern', 'false', 'final', 'float', 'for', 'friend', 'goto', 'if',
                      'inline', 'int', 'long', 'mutable', 'namespace', 'new', 'noexcept', 'not', 'not_eq', 'nullptr',
                      'operator', 'or', 'or_eq', 'override', 'private', 'protected', 'public', 'register',
                      'reinterpret_cast', 'return', 'short', 'signed', 'sizeof', 'static', 'static_assert',
                      'static_cast', 'struct', 'switch', 'template', 'this', 'thread_local', 'throw', 'true', 'try',
                      'typedef', 'typeid', 'typename', 'union', 'unsigned', 'using', 'virtual', 'void', 'volatile',
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL'])

# Library functions to preserve (Windows API, sockets, etc.)
library_functions = frozenset([
    'WSAStartup', 'socket', 'bind', 'listen', 'accept', 'recv', 'closesocket', 'WSACleanup',
    'memset', 'htons', 'MAKEWORD'  # Add more as needed
])

# Holds known non-user-defined functions
main_set = frozenset({'main'})
# Arguments in main
main_args = frozenset({'argc', 'argv'})

# Initialize libclang
index = Index.create()

# Normalize type names (remove spaces, pointers, references, etc.)
def normalize_type(type_str):
    type_str = re.sub(r'\s+', '', type_str)  # Remove spaces
    type_str = re.sub(r'[\*\&]', '', type_str)  # Remove pointers/references
    return type_str

# Extract function information using libclang
def extract_function_info(code):
    tu = index.parse('temp.cpp', args=['-std=c++17'], unsaved_files=[('temp.cpp', code)])
    func_info = {'return_type': '', 'name': '', 'params': [], 'body': ''}

    def traverse(node):
        if node.kind == CursorKind.FUNCTION_DECL:
            func_info['return_type'] = node.result_type.spelling
            func_info['name'] = node.spelling
            func_info['body'] = ''# node.extent.source_range.text
            for child in node.get_children():
                if child.kind == CursorKind.PARM_DECL:
                    param_type = child.type.spelling
                    param_name = child.spelling
                    if param_name:
                        func_info['params'].append((param_type, param_name))

    for node in tu.cursor.get_children():
        traverse(node)
    
    return func_info

def clean_gadget(gadget):
    fun_symbols = {}
    var_symbols = {}
    var_count = 1

    # Remove comments
    gadget = re.sub(r'/\*.*?\*/', '', gadget, flags=re.DOTALL)
    lines = gadget.split('\n')
    cleaned_gadget = ''
    for line in lines:
        pattern = r"^[ \t\r\n]*$"
        if re.fullmatch(pattern, line):
            continue
        line = re.sub(r'//.*', '', line)
        cleaned_gadget += line + '\n'

    # Extract function info
    func = extract_function_info(cleaned_gadget)
    if not func['name']:
        return cleaned_gadget  # Return unchanged if no function found

    result = cleaned_gadget

    # Normalize function name
    if func['name'] not in main_set and func['name'] not in keywords and func['name'] not in library_functions:
        normalized_return = normalize_type(func['return_type'])
        normalized_name = f"{normalized_return}_func_1"
        fun_symbols[func['name']] = normalized_name
        # Replace function name (only when followed by '(')
        result = re.sub(r'\b' + re.escape(func['name']) + r'\b(?=\s*\()', normalized_name, result)

    # Normalize parameters
    for param_type, param_name in func['params']:
        if param_name not in keywords and param_name not in main_args:
            normalized_param_type = normalize_type(param_type)
            normalized_param = f"{normalized_param_type}_var_{var_count}"
            var_symbols[param_name] = normalized_param
            var_count += 1
            # Replace parameter name in body and parameter list
            result = re.sub(r'\b' + re.escape(param_name) + r'\b(?!\s*\()', normalized_param, result)

    # Normalize variables in function body
    index = Index.create()
    tu = index.parse('temp.cpp', args=['-std=c++17'], unsaved_files=[('temp.cpp', func['body'])])
    def traverse_body(node):
        nonlocal var_count
        if node.kind == CursorKind.VAR_DECL:
            var_type = node.type.spelling
            var_name = node.spelling
            if var_name and var_name not in keywords and var_name not in main_args:
                normalized_var_type = normalize_type(var_type)
                normalized_var = f"{normalized_var_type}_var_{var_count}"
                var_symbols[var_name] = normalized_var
                var_count += 1
                nonlocal result
                result = re.sub(r'\b' + re.escape(var_name) + r'\b(?!\s*\()', normalized_var, result)
        for child in node.get_children():
            traverse_body(child)
    
    for node in tu.cursor.get_children():
        traverse_body(node)

    # Remove empty lines and trailing whitespace
    final_result = ''
    for line in result.split('\n'):
        if not re.fullmatch(r'^[ \t\r\n]*$', line):
            final_result += line.strip() + '\n'

    return final_result

class CXXNormalization:
    def __init__(self):
        pass

    def normalization(self, source):
        return clean_gadget(source)

    def normalization_df(self, df: DataFrame):
        df['code'] = df['code'].apply(lambda x: self.normalization(x))
        return df

if __name__ == '__main__':
    gadget = """
    void bad()
    {
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
    }
    """

    normalizer = CXXNormalization()
    nor_code = normalizer.normalization(gadget)
    print(nor_code)