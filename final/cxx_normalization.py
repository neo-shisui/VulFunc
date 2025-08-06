# Credit:
# - https://github.com/XUPT-SSS/TrVD/blob/main/clean_gadget.py

import re
import tree_sitter
import tree_sitter_cpp
from pandas import DataFrame

sockets = frozenset(['socket', 'bind', 'listen', 'accept', 'connect', 'send', 'recv', 'sendto', 'recvfrom', 'closesocket',
                     'WSAStartup', 'WSACleanup', 'getsockname', 'getpeername', 'getsockopt', 'setsockopt'])

# keywords up to C11 and C++17; immutable set
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
                      'wchar_t', 'while', 'xor', 'xor_eq', 'NULL', 'StrNCat', 'getaddrinfo', '_ui64toa', 'fclose',
                      'pthread_mutex_lock', 'gets_s', 'sleep', '_ui64tot', 'freopen_s', '_ui64tow', 'send', 'lstrcat',
                      'HMAC_Update', '__fxstat', 'StrCatBuff', '_mbscat', '_mbstok_s', '_cprintf_s',
                      'ldap_search_init_page', 'memmove_s', 'ctime_s', 'vswprintf', 'vswprintf_s', '_snwprintf',
                      '_gmtime_s', '_tccpy', '*RC6*', '_mbslwr_s', 'random', '__wcstof_internal', '_wcslwr_s',
                      '_ctime32_s', 'wcsncat*', 'MD5_Init', '_ultoa', 'snprintf', 'memset', 'syslog', '_vsnprintf_s',
                      'HeapAlloc', 'pthread_mutex_destroy', 'ChangeWindowMessageFilter', '_ultot', 'crypt_r',
                      '_strupr_s_l', 'LoadLibraryExA', '_strerror_s', 'LoadLibraryExW', 'wvsprintf', 'MoveFileEx',
                      '_strdate_s', 'SHA1', 'sprintfW', 'StrCatNW', '_scanf_s_l', 'pthread_attr_init', '_wtmpnam_s',
                      'snscanf', '_sprintf_s_l', 'dlopen', 'sprintfA', 'timed_mutex', 'OemToCharA', 'ldap_delete_ext',
                      'sethostid', 'popen', 'OemToCharW', '_gettws', 'vfork', '_wcsnset_s_l', 'sendmsg', '_mbsncat',
                      'wvnsprintfA', 'HeapFree', '_wcserror_s', 'realloc', '_snprintf*', 'wcstok', '_strncat*',
                      'StrNCpy', '_wasctime_s', 'push*', '_lfind_s', 'CC_SHA512', 'ldap_compare_ext_s', 'wcscat_s',
                      'strdup', '_chsize_s', 'sprintf_s', 'CC_MD4_Init', 'wcsncpy', '_wfreopen_s', '_wcsupr_s',
                      '_searchenv_s', 'ldap_modify_ext_s', '_wsplitpath', 'CC_SHA384_Final', 'MD2', 'RtlCopyMemory',
                      'lstrcatW', 'MD4', 'MD5', '_wcstok_s_l', '_vsnwprintf_s', 'ldap_modify_s', 'strerror',
                      '_lsearch_s', '_mbsnbcat_s', '_wsplitpath_s', 'MD4_Update', '_mbccpy_s', '_strncpy_s_l',
                      '_snprintf_s', 'CC_SHA512_Init', 'fwscanf_s', '_snwprintf_s', 'CC_SHA1', 'swprintf', 'fprintf',
                      'EVP_DigestInit_ex', 'strlen', 'SHA1_Init', 'strncat', '_getws_s', 'CC_MD4_Final', 'wnsprintfW',
                      'lcong48', 'lrand48', 'write', 'HMAC_Init', '_wfopen_s', 'wmemchr', '_tmakepath', 'wnsprintfA',
                      'lstrcpynW', 'scanf_s', '_mbsncpy_s_l', '_localtime64_s', 'fstream.open', '_wmakepath',
                      'Connection.open', '_tccat', 'valloc', 'setgroups', 'unlink', 'fstream.put', 'wsprintfA',
                      '*SHA1*', '_wsearchenv_s', 'ualstrcpyA', 'CC_MD5_Update', 'strerror_s', 'HeapCreate',
                      'ualstrcpyW', '__xstat', '_wmktemp_s', 'StrCatChainW', 'ldap_search_st', '_mbstowcs_s_l',
                      'ldap_modify_ext', '_mbsset_s', 'strncpy_s', 'move', 'execle', 'StrCat', 'xrealloc', 'wcsncpy_s',
                      '_tcsncpy*', 'execlp', 'RIPEMD160_Final', 'ldap_search_s', 'EnterCriticalSection', '_wctomb_s_l',
                      'fwrite', '_gmtime64_s', 'sscanf_s', 'wcscat', '_strupr_s', 'wcrtomb_s', 'VirtualLock',
                      'ldap_add_ext_s', '_mbscpy', '_localtime32_s', 'lstrcpy', '_wcsncpy*', 'CC_SHA1_Init', '_getts',
                      '_wfopen', '__xstat64', 'strcoll', '_fwscanf_s_l', '_mbslwr_s_l', 'RegOpenKey', 'makepath',
                      'seed48', 'CC_SHA256', 'sendto', 'execv', 'CalculateDigest', 'memchr', '_mbscpy_s', '_strtime_s',
                      'ldap_search_ext_s', '_chmod', 'flock', '__fxstat64', '_vsntprintf', 'CC_SHA256_Init', '_itoa_s',
                      '__wcserror_s', '_gcvt_s', 'fstream.write', 'sprintf', 'recursive_mutex', 'strrchr',
                      'gethostbyaddr', '_wcsupr_s_l', 'strcspn', 'MD5_Final', 'asprintf', '_wcstombs_s_l', '_tcstok',
                      'free', 'MD2_Final', 'asctime_s', '_alloca', '_wputenv_s', '_wcsset_s', '_wcslwr_s_l',
                      'SHA1_Update', 'filebuf.sputc', 'filebuf.sputn', 'SQLConnect', 'ldap_compare', 'mbstowcs_s',
                      'HMAC_Final', 'pthread_condattr_init', '_ultow_s', 'rand', 'ofstream.put', 'CC_SHA224_Final',
                      'lstrcpynA', 'bcopy', 'system', 'CreateFile*', 'wcscpy_s', '_mbsnbcpy*', 'open', '_vsnwprintf',
                      'strncpy', 'getopt_long', 'CC_SHA512_Final', '_vsprintf_s_l', 'scanf', 'mkdir', '_localtime_s',
                      '_snprintf', '_mbccpy_s_l', 'memcmp', 'final', '_ultoa_s', 'lstrcpyW', 'LoadModule',
                      '_swprintf_s_l', 'MD5_Update', '_mbsnset_s_l', '_wstrtime_s', '_strnset_s', 'lstrcpyA',
                      '_mbsnbcpy_s', 'mlock', 'IsBadHugeWritePtr', 'copy', '_mbsnbcpy_s_l', 'wnsprintf', 'wcscpy',
                      'ShellExecute', 'CC_MD4', '_ultow', '_vsnwprintf_s_l', 'lstrcpyn', 'CC_SHA1_Final', 'vsnprintf',
                      '_mbsnbset_s', '_i64tow', 'SHA256_Init', 'wvnsprintf', 'RegCreateKey', 'strtok_s', '_wctime32_s',
                      '_i64toa', 'CC_MD5_Final', 'wmemcpy', 'WinExec', 'CreateDirectory*', 'CC_SHA256_Update',
                      '_vsnprintf_s_l', 'jrand48', 'wsprintf', 'ldap_rename_ext_s', 'filebuf.open', '_wsystem',
                      'SHA256_Update', '_cwscanf_s', 'wsprintfW', '_sntscanf', '_splitpath', 'fscanf_s', 'strpbrk',
                      'wcstombs_s', 'wscanf', '_mbsnbcat_s_l', 'strcpynA', 'pthread_cond_init', 'wcsrtombs_s',
                      '_wsopen_s', 'CharToOemBuffA', 'RIPEMD160_Update', '_tscanf', 'HMAC', 'StrCCpy',
                      'Connection.connect', 'lstrcatn', '_mbstok', '_mbsncpy', 'CC_SHA384_Update', 'create_directories',
                      'pthread_mutex_unlock', 'CFile.Open', 'connect', '_vswprintf_s_l', '_snscanf_s_l', 'fputc',
                      '_wscanf_s', '_snprintf_s_l', 'strtok', '_strtok_s_l', 'lstrcatA', 'snwscanf',
                      'pthread_mutex_init', 'fputs', 'CC_SHA384_Init', '_putenv_s', 'CharToOemBuffW',
                      'pthread_mutex_trylock', '__wcstoul_internal', '_memccpy', '_snwprintf_s_l', '_strncpy*',
                      'wmemset', 'MD4_Init', '*RC4*', 'strcpyW', '_ecvt_s', 'memcpy_s', 'erand48', 'IsBadHugeReadPtr',
                      'strcpyA', 'HeapReAlloc', 'memcpy', 'ldap_rename_ext', 'fopen_s', 'srandom', '_cgetws_s',
                      '_makepath', 'SHA256_Final', 'remove', '_mbsupr_s', 'pthread_mutexattr_init',
                      '__wcstold_internal', 'StrCpy', 'ldap_delete', 'wmemmove_s', '_mkdir', 'strcat', '_cscanf_s_l',
                      'StrCAdd', 'swprintf_s', '_strnset_s_l', 'close', 'ldap_delete_ext_s', 'ldap_modrdn', 'strchr',
                      '_gmtime32_s', '_ftcscat', 'lstrcatnA', '_tcsncat', 'OemToChar', 'mutex', 'CharToOem', 'strcpy_s',
                      'lstrcatnW', '_wscanf_s_l', '__lxstat64', 'memalign', 'MD2_Init', 'StrCatBuffW', 'StrCpyN',
                      'CC_MD5', 'StrCpyA', 'StrCatBuffA', 'StrCpyW', 'tmpnam_r', '_vsnprintf', 'strcatA', 'StrCpyNW',
                      '_mbsnbset_s_l', 'EVP_DigestInit', '_stscanf', 'CC_MD2', '_tcscat', 'StrCpyNA', 'xmalloc',
                      '_tcslen', '*MD4*', 'vasprintf', 'strxfrm', 'chmod', 'ldap_add_ext', 'alloca', '_snscanf_s',
                      'IsBadWritePtr', 'swscanf_s', 'wmemcpy_s', '_itoa', '_ui64toa_s', 'EVP_DigestUpdate',
                      '__wcstol_internal', '_itow', 'StrNCatW', 'strncat_s', 'ualstrcpy', 'execvp', '_mbccat',
                      'EVP_MD_CTX_init', 'assert', 'ofstream.write', 'ldap_add', '_sscanf_s_l', 'drand48', 'CharToOemW',
                      'swscanf', '_itow_s', 'RIPEMD160_Init', 'CopyMemory', 'initstate', 'getpwuid', 'vsprintf',
                      '_fcvt_s', 'CharToOemA', 'setuid', 'malloc', 'StrCatNA', 'strcat_s', 'srand', 'getwd',
                      '_controlfp_s', 'olestrcpy', '__wcstod_internal', '_mbsnbcat', 'lstrncat', 'des_*',
                      'CC_SHA224_Init', 'set*', 'vsprintf_s', 'SHA1_Final', '_umask_s', 'gets', 'setstate',
                      'wvsprintfW', 'LoadLibraryEx', 'ofstream.open', 'calloc', '_mbstrlen', '_cgets_s', '_sopen_s',
                      'IsBadStringPtr', 'wcsncat_s', 'add*', 'nrand48', 'create_directory', 'ldap_search_ext',
                      '_i64toa_s', '_ltoa_s', '_cwscanf_s_l', 'wmemcmp', '__lxstat', 'lstrlen',
                      'pthread_condattr_destroy', '_ftcscpy', 'wcstok_s', '__xmknod', 'pthread_attr_destroy',
                      'sethostname', '_fscanf_s_l', 'StrCatN', 'RegEnumKey', '_tcsncpy', 'strcatW', 'AfxLoadLibrary',
                      'setenv', 'tmpnam', '_mbsncat_s_l', '_wstrdate_s', '_wctime64_s', '_i64tow_s', 'CC_MD4_Update',
                      'ldap_add_s', '_umask', 'CC_SHA1_Update', '_wcsset_s_l', '_mbsupr_s_l', 'strstr', '_tsplitpath',
                      'memmove', '_tcscpy', 'vsnprintf_s', 'strcmp', 'wvnsprintfW', 'tmpfile', 'ldap_modify',
                      '_mbsncat*', 'mrand48', 'sizeof', 'StrCatA', '_ltow_s', '*desencrypt*', 'StrCatW', '_mbccpy',
                      'CC_MD2_Init', 'RIPEMD160', 'ldap_search', 'CC_SHA224', 'mbsrtowcs_s', 'update', 'ldap_delete_s',
                      'getnameinfo', '*RC5*', '_wcsncat_s_l', 'DriverManager.getConnection', 'socket', '_cscanf_s',
                      'ldap_modrdn_s', '_wopen', 'CC_SHA256_Final', '_snwprintf*', 'MD2_Update', 'strcpy',
                      '_strncat_s_l', 'CC_MD5_Init', 'mbscpy', 'wmemmove', 'LoadLibraryW', '_mbslen', '*alloc',
                      '_mbsncat_s', 'LoadLibraryA', 'fopen', 'StrLen', 'delete', '_splitpath_s',
                      'CreateFileTransacted*', 'MD4_Final', '_open', 'CC_SHA384', 'wcslen', 'wcsncat', '_mktemp_s',
                      'pthread_mutexattr_destroy', '_snwscanf_s', '_strset_s', '_wcsncpy_s_l', 'CC_MD2_Final',
                      '_mbstok_s_l', 'wctomb_s', 'MySQL_Driver.connect', '_snwscanf_s_l', '*_des_*', 'LoadLibrary',
                      '_swscanf_s_l', 'ldap_compare_s', 'ldap_compare_ext', '_strlwr_s', 'GetEnvironmentVariable',
                      'cuserid', '_mbscat_s', 'strspn', '_mbsncpy_s', 'ldap_modrdn2', 'LeaveCriticalSection',
                      'CopyFile', 'getpwd', 'sscanf', 'creat', 'RegSetValue', 'ldap_modrdn2_s', 'CFile.Close',
                      '*SHA_1*', 'pthread_cond_destroy', 'CC_SHA512_Update', '*RC2*', 'StrNCatA', '_mbsnbcpy',
                      '_mbsnset_s', 'crypt', 'excel', '_vstprintf', 'xstrdup', 'wvsprintfA', 'getopt', 'mkstemp',
                      '_wcsnset_s', '_stprintf', '_sntprintf', 'tmpfile_s', 'OpenDocumentFile', '_mbsset_s_l',
                      '_strset_s_l', '_strlwr_s_l', 'ifstream.open', 'xcalloc', 'StrNCpyA', '_wctime_s',
                      'CC_SHA224_Update', '_ctime64_s', 'MoveFile', 'chown', 'StrNCpyW', 'IsBadReadPtr', '_ui64tow_s',
                      'IsBadCodePtr', 'getc', 'OracleCommand.ExecuteOracleScalar', 'AccessDataSource.Insert',
                      'IDbDataAdapter.FillSchema', 'IDbDataAdapter.Update', 'GetWindowText*', 'SendMessage',
                      'SqlCommand.ExecuteNonQuery', 'streambuf.sgetc', 'streambuf.sgetn', 'OracleCommand.ExecuteScalar',
                      'SqlDataSource.Update', '_Read_s', 'IDataAdapter.Fill', '_wgetenv', '_RecordsetPtr.Open*',
                      'AccessDataSource.Delete', 'Recordset.Open*', 'filebuf.sbumpc', 'DDX_*', 'RegGetValue',
                      'fstream.read*', 'SqlCeCommand.ExecuteResultSet', 'SqlCommand.ExecuteXmlReader', 'main',
                      'streambuf.sputbackc', 'read', 'm_lpCmdLine', 'CRichEditCtrl.Get*', 'istream.putback',
                      'SqlCeCommand.ExecuteXmlReader', 'SqlCeCommand.BeginExecuteXmlReader', 'filebuf.sgetn',
                      'OdbcDataAdapter.Update', 'filebuf.sgetc', 'SQLPutData', 'recvfrom',
                      'OleDbDataAdapter.FillSchema', 'IDataAdapter.FillSchema', 'CRichEditCtrl.GetLine',
                      'DbDataAdapter.Update', 'SqlCommand.ExecuteReader', 'istream.get', 'ReceiveFrom', '_main',
                      'fgetc', 'DbDataAdapter.FillSchema', 'kbhit', 'UpdateCommand.Execute*', 'Statement.execute',
                      'fgets', 'SelectCommand.Execute*', 'getch', 'OdbcCommand.ExecuteNonQuery', 'CDaoQueryDef.Execute',
                      'fstream.getline', 'ifstream.getline', 'SqlDataAdapter.FillSchema', 'OleDbCommand.ExecuteReader',
                      'Statement.execute*', 'SqlCeCommand.BeginExecuteNonQuery', 'OdbcCommand.ExecuteScalar',
                      'SqlCeDataAdapter.Update', 'sendmessage', 'mysqlpp.DBDriver', 'fstream.peek', 'Receive',
                      'CDaoRecordset.Open', 'OdbcDataAdapter.FillSchema', '_wgetenv_s', 'OleDbDataAdapter.Update',
                      'readsome', 'SqlCommand.BeginExecuteXmlReader', 'recv', 'ifstream.peek', '_Main', '_tmain',
                      '_Readsome_s', 'SqlCeCommand.ExecuteReader', 'OleDbCommand.ExecuteNonQuery', 'fstream.get',
                      'IDbCommand.ExecuteScalar', 'filebuf.sputbackc', 'IDataAdapter.Update', 'streambuf.sbumpc',
                      'InsertCommand.Execute*', 'RegQueryValue', 'IDbCommand.ExecuteReader', 'SqlPipe.ExecuteAndSend',
                      'Connection.Execute*', 'getdlgtext', 'ReceiveFromEx', 'SqlDataAdapter.Update', 'RegQueryValueEx',
                      'SQLExecute', 'pread', 'SqlCommand.BeginExecuteReader', 'AfxWinMain', 'getchar',
                      'istream.getline', 'SqlCeDataAdapter.Fill', 'OleDbDataReader.ExecuteReader',
                      'SqlDataSource.Insert', 'istream.peek', 'SendMessageCallback', 'ifstream.read*',
                      'SqlDataSource.Select', 'SqlCommand.ExecuteScalar', 'SqlDataAdapter.Fill',
                      'SqlCommand.BeginExecuteNonQuery', 'getche', 'SqlCeCommand.BeginExecuteReader', 'getenv',
                      'streambuf.snextc', 'Command.Execute*', '_CommandPtr.Execute*', 'SendNotifyMessage',
                      'OdbcDataAdapter.Fill', 'AccessDataSource.Update', 'fscanf', 'QSqlQuery.execBatch',
                      'DbDataAdapter.Fill', 'cin', 'DeleteCommand.Execute*', 'QSqlQuery.exec', 'PostMessage',
                      'ifstream.get', 'filebuf.snextc', 'IDbCommand.ExecuteNonQuery', 'Winmain', 'fread', 'getpass',
                      'GetDlgItemTextCCheckListBox.GetCheck', 'DISP_PROPERTY_EX', 'pread64', 'Socket.Receive*',
                      'SACommand.Execute*', 'SQLExecDirect', 'SqlCeDataAdapter.FillSchema', 'DISP_FUNCTION',
                      'OracleCommand.ExecuteNonQuery', 'CEdit.GetLine', 'OdbcCommand.ExecuteReader', 'CEdit.Get*',
                      'AccessDataSource.Select', 'OracleCommand.ExecuteReader', 'OCIStmtExecute', 'getenv_s',
                      'DB2Command.Execute*', 'OracleDataAdapter.FillSchema', 'OracleDataAdapter.Fill', 'CComboBox.Get*',
                      'SqlCeCommand.ExecuteNonQuery', 'OracleCommand.ExecuteOracleNonQuery', 'mysqlpp.Query',
                      'istream.read*', 'CListBox.GetText', 'SqlCeCommand.ExecuteScalar', 'ifstream.putback', 'readlink',
                      'CHtmlEditCtrl.GetDHtmlDocument', 'PostThreadMessage', 'CListCtrl.GetItemText',
                      'OracleDataAdapter.Update', 'OleDbCommand.ExecuteScalar', 'stdin', 'SqlDataSource.Delete',
                      'OleDbDataAdapter.Fill', 'fstream.putback', 'IDbDataAdapter.Fill', '_wspawnl', 'fwprintf',
                      'sem_wait', '_unlink', 'ldap_search_ext_sW', 'signal', 'PQclear', 'PQfinish', 'PQexec',
                      'PQresultStatus','ifdef','endif','bool','void',

                      # Socket functions
                      'WSAStartup', 'WSACleanup',
                      ])
# holds known non-user-defined functions; immutable set
main_set = frozenset({'main'})
# arguments in main d2a; immutable set
main_args = frozenset({'argc', 'argv'})

 # Initialize the tree-sitter parser for C++
CPP_LANGUAGE = tree_sitter.Language(tree_sitter_cpp.language())
parser = tree_sitter.Parser()
parser.language = CPP_LANGUAGE

# Get struct/class/enum from C++ gadget by type `type_identifier` (using tree-sitter)
def get_struct_class(node, types: list):
    token = node.text.decode('utf8')
    if node.type == 'type_identifier' and token not in types:
        return types.append(token)

    # Get the children of the node
    children = node.children
    for child in children:
        # Recursively call the function for each child
        get_struct_class(child, types)

def get_function_type(node, functions: dict, variables: dict):
    if node.type == 'function_definition':
        function_signature = node.text.decode('utf8').split('(')[0].strip()  # Get the function signature before the parameters
        # print('Function Signature: ', function_signature)

        function_name = function_signature.split(' ')[-1]  # Get the function name from the signature
        return_type = normalize_type(' '.join(function_signature.split(' ')[:-1]))
        # print('Function Name: ', function_name, ' Return Type: ', return_type)
        if function_name not in functions:
            functions[function_name] = return_type
    elif node.type == 'assignment_expression' and b'(' in node.text:  # Check if the node is an assignment expression with a function call
        expression = node.text.decode('utf8')
        var_expr = expression.split('=')[0].strip()  # Get the variable expression before the assignment
        func_expr = expression.split('=')[1].split('(')[0].strip()  # Get the function expression after the assignment
        if var_expr in variables:
            functions[func_expr] = variables[var_expr]  # Use the variable type as the function type

    # Recursively traverse the children of the node
    # print(node.type, node.text)
    # print('')
    for child in node.children:
        get_function_type(child, functions, variables)

def get_variable_type(node, variables: dict):
    if node.type == 'declaration':
        declare_expr = node.text.decode('utf8').split('=')[0].rstrip().replace(';', '')  # Get the declaration expression, remove assignment and semicolon
        # print('Declaration: ', declare_expr)

        _type = ' '.join(declare_expr.split(' ')[:-1])  # Get the type from the declaration
        variable_name = declare_expr.split(' ')[-1]  # Get the variable name from the declaration

        # Check if variable is array
        if '[' in variable_name and ']' in variable_name:
            variable_name = variable_name.split('[')[0]
            _type += '[]'  # Append array notation to type

        # print('Variable Name: ', variable_name, ' Type: ', _type)
        # print('')

        # Normalize the type and store it in the variables dictionary
        normalized_type = normalize_type(_type)
        if variable_name not in variables:
            variables[variable_name] = normalized_type

        # print(node.text)
        _type = ''
        for child in node.children:
            # print(child.type, child.text)
            # Pointer or reference
            if child.type == 'pointer_declarator':
                for subchild in child.children:
                    pass
                    # print('Pointer: ', subchild.type, subchild.text)
            # Get type
            if child.type == 'type_identifier' or child.type == 'primitive_type' or child.type == 'struct_specifier' or child.type == 'template_type':
                _type = normalize_type(child.text.decode('utf8'))
                # print('Type: ', child.text, ' ', _type)
            # print(child.type, child.text)
            # if len(child.children) != 0:

        # token = node.text.decode('utf8')
        # if token not in variables:
        #     variables[token] = []
    else:
        # Recursively traverse the children of the node
        for child in node.children:
            # If the child is a type identifier, add it to the variables dictionary
            # if child.type == 'type_identifier':
            #     token = child.text.decode('utf8')
            #     if token not in variables:
            #         variables[token] = []
            # Recursively call the function for each child
            get_variable_type(child, variables)

# Normalize type names (remove spaces, pointers, etc.)
def normalize_type(type_str):
    # Replace multiple spaces with a single underscore
    type_str = re.sub(r'\s+', '_', type_str) 

    # Replace pointers by 'ptr' and references by 'ref'
    type_str = re.sub(r'\*+', '_ptr', type_str)  # Replace multiple pointers with 'ptr'
    type_str = re.sub(r'\&+', '_ref', type_str)

    # Replace pattern <...> with 'template_<type>'
    type_str = re.sub(r'<([^>]*)>', r'_template_\1', type_str)

    # Replace array notation with 'array'
    type_str = re.sub(r'\[\s*\]', '_array', type_str)

    # Remove multi underscores by single underscore
    type_str = re.sub(r'_+', '_', type_str)

    return type_str

# Extract function information using tree-sitter
def extract_functions(code):
    tree = parser.parse(bytes(code, 'utf8'))
    root = tree.root_node
    functions = []

    def traverse(node):
        if node.type == 'function_definition':
            return_type = ''
            func_name = ''
            params = []
            body = ''
            
            for child in node.children:
                if child.type == 'type_identifier' or child.type == 'primitive_type' or child.type == 'type_qualifier':
                    return_type += child.text.decode('utf8') + ' '
                elif child.type == 'function_declarator':
                    for subchild in child.children:
                        if subchild.type == 'identifier':
                            func_name = subchild.text.decode('utf8')
                        elif subchild.type == 'parameter_list':
                            for param in subchild.children:
                                if param.type == 'parameter_declaration':
                                    param_type = ''
                                    param_name = ''
                                    for pchild in param.children:
                                        if pchild.type in ('type_identifier', 'primitive_type', 'type_qualifier'):
                                            param_type += pchild.text.decode('utf8') + ' '
                                        elif pchild.type == 'identifier':
                                            param_name = pchild.text.decode('utf8')
                                    if param_name:
                                        params.append((param_type.strip(), param_name))
                elif child.type == 'compound_statement':
                    body = child.text.decode('utf8')
            
            if func_name:
                functions.append({
                    'return_type': return_type.strip(),
                    'name': func_name,
                    'params': params,
                    'body': body
                })
        
        for child in node.children:
            traverse(child)

    traverse(root)
    return functions

# Function to clean invalid unicode escape sequences
def clean_invalid_escapes(text):
    # Replace invalid \x escapes with a placeholder or remove them
    cleaned = re.sub(r'\\x(?:[^0-9a-fA-F]|[0-9a-fA-F]{1,3}(?![0-9a-fA-F]))', '', text)
    cleaned = re.sub(r'\\U[0-9a-fA-F]{0,7}(?![0-9a-fA-F])', '', cleaned)
    return cleaned

# input is a C++ gadget as a string (function body)
def clean_gadget(gadget):
    # dictionary; map d2a name to symbol name + number
    fun_symbols = {}
    # dictionary; map variable name to symbol name + number
    var_symbols = {}
    # dictionary; map class/struct name to symbol name + number
    type_symbols = {}

    gadget = clean_invalid_escapes(gadget)

    try:
        tree = parser.parse(gadget.encode('utf-8').decode('unicode_escape').encode())
    except Exception as e:
        print(f"Error parsing gadget: {e}")
        return None

    vars_type = dict()
    get_variable_type(tree.root_node, vars_type)
    # print('Variable types:', vars_type)

    func_type = dict()
    get_function_type(tree.root_node, func_type, vars_type)
    # print('Function types:', func_type)

    fun_count = 1
    var_count = 1
    type_count = 1

    # Remove comments from the function
    # Remove multi-line comments
    gadget = re.sub('/\\*.*?\\*/', '', gadget, flags=re.DOTALL) 
    lines = gadget.split('\n')
    gadget = ''
    for line in lines:
        # Check if empty line
        pattern = r"^[ \t\r\n]*$"
        if re.fullmatch(pattern, line):
            continue
        
        # Remove single-line comments (content after //)
        line = re.sub('//.*', '', line)
        gadget += line + '\n'

    # Regular expression to catch multi-line comment
    rx_comment = re.compile(r'\*/\s*$')
    
    # Regular expression to find d2a name candidates
    # - followed by an opening parenthesis (
    rx_fun = re.compile(r'\b([_A-Za-z]\w*)\b(?=\s*\()')

    # Regular expression to find variable name candidates
    # - followed by a space and then an opening parenthesis (or no parenthesis)
    # - or not followed by a parenthesis and symbol < at all
    #rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?!\s*\()')
    rx_var = re.compile(r'\b([_A-Za-z]\w*)\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*[\(<])')

    # process function if not the header line and not a multi-line commented line
    if rx_comment.search(gadget) is None:
        # remove all string literals (keep the quotes)
        nostrlit_line = re.sub(r'".*?"', '""', gadget)
        # remove all character literals
        nocharlit_line = re.sub(r"'.*?'", "''", nostrlit_line)
        # replace any non-ASCII characters with empty string
        ascii_line = re.sub(r'[^\x00-\x7f]', r'', nocharlit_line)

        # return, in order, all regex matches at string list; preserves order for semantics
        user_fun = rx_fun.findall(ascii_line)
        user_var = rx_var.findall(ascii_line)
        user_type = []

        # get_struct_class(parser.parse(bytes(ascii_line, 'utf8')).root_node, user_type)
        # for var_name in user_var:
        #     if var_name in user_type:
        #         user_var.remove(var_name)

        # Could easily make a "clean gadget" type class to prevent duplicate functionality
        # of creating/comparing symbol names for functions and variables in much the same way.
        # The comparison frozenset, symbol dictionaries, and counters would be class scope.
        # So would only need to pass a string list and a string literal for symbol names to
        # another d2a.
        for fun_name in user_fun:
            # Check if function name is uppercase (e.g., a macro or constant)
            if fun_name.isupper():
                # If it is, we can skip it as it is likely a macro or constant
                continue

            if len({fun_name}.difference(main_set)) != 0 and len({fun_name}.difference(keywords)) != 0:
                # DEBUG
                #print('comparing ' + str(fun_name + ' to ' + str(main_set)))
                #print(fun_name + ' diff len from main is ' + str(len({fun_name}.difference(main_set))))
                #print('comparing ' + str(fun_name + ' to ' + str(keywords)))
                #print(fun_name + ' diff len from keywords is ' + str(len({fun_name}.difference(keywords))))
                ###
                # check to see if d2a name already in dictionary
                if fun_name not in fun_symbols.keys():
                    # Prepend return type to function name
                    if fun_name in func_type:
                        fun_symbols[fun_name] = 'func_' + func_type[fun_name] # + str(fun_count)
                    else:
                        fun_symbols[fun_name] = 'func_unk' #'func_' + str(fun_count)
                    # fun_count += 1
                # ensure that only d2a name gets replaced (no variable name with same
                # identifier); uses positive lookforward
                ascii_line = re.sub(r'\b(' + fun_name + r')\b(?=\s*\()', fun_symbols[fun_name], ascii_line)

        for type_name in user_type:
            if type_name not in type_symbols.keys():
                type_symbols[type_name] = 'type_' + str(type_count)
                type_count += 1
            # Replace type names
            ascii_line = re.sub(r'\b(' + type_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                type_symbols[type_name], ascii_line)
            ascii_line = re.sub(r'\b(' + type_name + r')\b(?!\s*\()', type_symbols[type_name], ascii_line)
        # print('Type symbols:', type_symbols)

        for var_name in user_var:
            # Check if variable is uppercase (e.g., a macro or constant)
            if var_name.isupper():
                # If it is, we can skip it as it is likely a macro or constant
                continue

            # next line is the nuanced difference between fun_name and var_name
            if len({var_name}.difference(keywords)) != 0 and len({var_name}.difference(main_args)) != 0 and var_name not in type_symbols.values():
                # DEBUG
                #print('comparing ' + str(var_name + ' to ' + str(keywords)))
                #print(var_name + ' diff len from keywords is ' + str(len({var_name}.difference(keywords))))
                #print('comparing ' + str(var_name + ' to ' + str(main_args)))
                #print(var_name + ' diff len from main args is ' + str(len({var_name}.difference(main_args))))
                ###
                # check to see if variable name already in dictionary
                if var_name not in var_symbols.keys() and var_name not in user_type:
                    # print('Adding variable: ' + var_name)
                    if var_name in vars_type:
                        var_symbols[var_name] = 'var_' + vars_type[var_name] #+ str(var_count)
                    else:
                        var_symbols[var_name] = 'var_unk' # + str(var_count)
                    var_count += 1
                    # print('Variable symbols:', var_name)
                else:
                    continue

                # ensure that only variable name gets replaced (no d2a name with same
                # identifier); uses negative lookforward
                ascii_line = re.sub(r'\b(' + var_name + r')\b(?:(?=\s*\w+\()|(?!\s*\w+))(?!\s*\()', \
                                    var_symbols[var_name], ascii_line)
    
    else:
        return None

    # Continue remove emtpty lines and trailing whitespace
    cleaned_gadget = ''
    lines = ascii_line.split('\n')
    for line in lines:
        # Check if empty line
        pattern = r"^[ \t\r\n]*$"
        if re.fullmatch(pattern, line):
            continue

        cleaned_gadget = cleaned_gadget + line.strip() + '\n'
    return cleaned_gadget

class CXXNormalization:
    def __init__(self):
        """Initialize the CXXNormalization class."""
        pass  # No initialization needed based on the provided function

    def normalization(self, source):
        """Normalize source code by removing comments and cleaning."""
        nor_code = clean_gadget(source)

        # nor_code = []
        # print(source)
        # for func in source['code']:
        #     print(func)
        #     code = clean_gadget(func)
        #     print("Func:", func)
        #     # print(code) 
        #     nor_code.append(code)
        return nor_code
    
    def normalization_df(self, df: DataFrame):
        """Normalize a DataFrame containing source code."""
        df['code'] = df['code'].apply(lambda x: self.normalization({'code': [x]})[0])
        print(df['code'][0])  # Print the first normalized code for debugging
        return df   

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

    # print(' ' * level + name)
    print(' ' * level + name + ' ' + token + '\n')
    text_tree = text_tree + ' ' * level + name + '\n'
    for child in children:
        print_ast(child, level + 1, text_tree)

if __name__ == '__main__':
    gadget = """
void bad()
{
    char x[] = "Hello, World!";
    char* data;
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
        }"""

    normalizer = CXXNormalization()
    nor_code = normalizer.normalization(gadget)
    print(nor_code)

    tree = parser.parse(gadget.encode('utf-8').decode('unicode_escape').encode())
    # print_ast(tree)

    vars_type = dict()
    get_variable_type(tree.root_node, vars_type)
    # print("Variable types found:", vars_type)

    functions = dict()
    get_function_type(tree.root_node, functions, vars_type)
    # print("Functions found:", functions)

    # types = []
    # get_struct_class(tree.root_node, types)
    # print("Types found:", types)
    #