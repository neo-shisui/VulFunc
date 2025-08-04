import os
import re
import json
import pandas as pd
from pathlib import Path
import pickle
import argparse
import xml.etree.ElementTree as ET
from clang.cindex import Index, CursorKind

# Initialize libclang
index = Index.create()

# def get_juliet_files(root_dir):
#     """
#     Recursively find all .c and .cpp files in the given directory.
    
#     Args:
#         root_dir (str): Path to the root directory of the Juliet Test Suite.
        
#     Returns:
#         list: List of absolute file paths to .c and .cpp files.
#     """
#     cpp_files = []
#     root_path = Path(root_dir)
    
#     # Walk through directory recursively
#     for file_path in root_path.rglob("*"):
#         if file_path.is_file() and file_path.suffix in (".c", ".cpp"):
#             cpp_files.append(str(file_path.absolute()))
    
#     return cpp_files

# # cpp_files = get_juliet_files("/home/shisui/Documents/Source/VulFunc/VulFunc/datasets/SARD/2017-10-01-juliet-test-suite-for-c-cplusplus-v1-3/C/testcases/")
# # print(cpp_files)

# def extract_cwe_and_filename(file_path):
#     """
#     Extract CWE ID and base filename from a file path.
    
#     Args:
#         file_path (str): Absolute path to the file.
#         root_dir (str): Root directory of the Juliet Test Suite.
        
#     Returns:
#         tuple: (cwe_id, base_filename)
#     """
    
#     filename = os.path.splitext(os.path.basename(file_path))[0]
#     cwe_id = filename.split("_")[0]
#     # Remove bonus suffix (e.g., 'a' in test_18a)
#     base_filename = re.sub(r'[a-z]$', '', filename)
    
#     return cwe_id, base_filename

# def extract_function_by_name(file_path, function_name):
#     func = None
#     try:
#         tu = index.parse(file_path)
#         for node in tu.cursor.walk_preorder():
#             if node.kind == CursorKind.FUNCTION_DECL and node.is_definition() and node.spelling == function_name:
#                 start_line = node.extent.start.line
#                 end_line = node.extent.end.line
#                 with open(file_path, 'r') as f:
#                     lines = f.readlines()[start_line-1:end_line]
#                 func = ''.join(lines).strip()
#     except Exception as e:
#         print(f"Error parsing {file_path}: {e}")
#     return func

# def extract_function_calls(code):
#     """
#     Extract function call names from a function's code.
    
#     Args:
#         code (str): Source code of the function.
        
#     Returns:
#         list: List of function names called in the code.
#     """
#     # Simple regex to match function calls (e.g., goodB2G1(), goodB2G2();)
#     pattern = r'\b(\w+)\s*\(\s*\)\s*;'
#     matches = re.findall(pattern, code)
#     return matches

# def extract_functions_from_cpp(file_path, base_filename):
#     """
#     Extract functions from a C++ file and label them as good (0) or bad (1).
    
#     Args:
#         file_path (str): Path to the C++ file.
#         base_filename (str): Base filename without bonus suffix (e.g., 'test_18').
        
#     Returns:
#         list: List of dictionaries with function code and target (0 or 1).
#     """
#     functions = []
#     temp_functions = dict()
#     try:
#         tu = index.parse(file_path)
#         for node in tu.cursor.walk_preorder():
#             if node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
#                 func_name = node.spelling
#                 start_line = node.extent.start.line
#                 end_line = node.extent.end.line
#                 with open(file_path, 'r') as f:
#                     lines = f.readlines()[start_line-1:end_line]
#                 code = ''.join(lines).strip()
                
#                 # Determine if function is good or bad
#                 target = 0  # Default to good
#                 if f"{base_filename}_bad" in func_name:
#                     target = 1
#                 elif f"{base_filename}_badSource" in func_name:
#                     print(f"{base_filename}_badSource")
#                     target = 1
#                 elif f"{base_filename}_good" in func_name:
#                     funcs = extract_function_calls(code)
#                     for func in funcs:
#                         # print(func)
#                         functions.append({
#                             "code": temp_functions[func],
#                             "target": 0
#                         })
#                     continue
#                 elif f"{base_filename}_good" not in func_name:
#                     # print(func_name)
#                     temp_functions.update({func_name: code})
#                     continue  # Skip functions that are neither good nor bad
                
#                 # print(func_name)
#                 functions.append({
#                     "code": code,
#                     "target": target
#                 })
#     except Exception as e:
#         print(f"Error parsing {file_path}: {e}")
#     return functions

# def create_sard_json(root_dir, output_json):
#     """
#     Create a JSON file with CWE, target, and function code from Juliet Test Suite.

#     Args:
#         root_dir (str): Path to the Juliet Test Suite directory.
#         output_json (str): Path to save the output JSON file.
#     """
#     dataset = []
#     cpp_files = get_juliet_files(root_dir)
#     print(f"Found {len(cpp_files)} C/C++ files in the Juliet Test Suite.")
#     i = 0
#     for file_path in cpp_files:
#         cwe_id, base_filename = extract_cwe_and_filename(file_path)
#         # print(cwe_id)
#         # print(base_filename)
#         functions = extract_functions_from_cpp(file_path, base_filename)

#         for func in functions:
#             dataset.append({
#                 "cwd": cwe_id,
#                 "target": func["target"],
#                 "code": func["code"]
#             })
#         if i % 1000 == 0 and i != 0:
#             print("[+] Progress: ", i)
#             # break
#         # if i == 0: 
#         #     break
#         i = i + 1
    
#     # Convert list to pandas DataFrame
#     dataset_df = pd.DataFrame(dataset)
    
#     # Verify DataFrame type
#     if not isinstance(dataset_df, pd.core.frame.DataFrame):
#         raise TypeError("Dataset is not a pandas.core.frame.DataFrame")

#     with open(output_json, 'w') as f:
#         json.dump(dataset_df.to_dict(orient='records'), f)
#     print(f"Created {output_json} with {len(dataset)} functions.")

class CXXFunctionParser:
    """
    Parses C++ functions from a source file and extracts their code and target labels.
    """
    
    def __init__(self):
        self.functions = []

    def extract_functions_from_cpp(self, file_path):
        """
        Extract functions from a C/C++ file
        
        Args:
            file_path (str): Path to the C/C++ file.
            
        Returns:
            list functions
        """
        functions = []
        try:
            tu = index.parse(file_path)
            for node in tu.cursor.walk_preorder():
                if node.kind == CursorKind.FUNCTION_DECL and node.is_definition():
                    func_name = node.spelling
                    start_line = node.extent.start.line
                    end_line = node.extent.end.line
                    with open(file_path, 'r') as f:
                        lines = f.readlines()[start_line-1:end_line]
                    code = ''.join(lines).strip()
                    
                    functions.append({
                        "name": func_name,
                        "code": code
                    })
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        return functions


# Example usage
if __name__ == "__main__":
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Cxx Function Parser')
    parser.add_argument('--file', type=str, required=True, help='Path to the C++ file to parse')
    parser.add_argument('--output', type=str, required=False, default='functions.json', help='Path to the output JSON file')
    args = parser.parse_args()

    # Create a CXXFunctionParser instance
    parser = CXXFunctionParser()
    functions = parser.extract_functions_from_cpp(args.file)

    # Save the extracted functions to a JSON file
    with open(args.output, 'w') as f:
        json.dump(functions, f, indent=4)
    
    print(f"Extracted {len(functions)} functions from {args.file} and saved to {args.output}.")