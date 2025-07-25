import re
import pandas as pd
from pandas import DataFrame
try:
    from clean_gadget import clean_gadget
except:
    from preprocess.clean_gadget import clean_gadget

class CXXNormalization:
    def __init__(self):
        """Initialize the CXXNormalization class."""
        pass  # No initialization needed based on the provided function

    def normalization(self, source):
        """Normalize source code by removing comments and cleaning."""
        nor_code = []
        for fun in source['code']:
            lines = fun.split('\n')
            code = ''
            for line in lines:
                line = line.strip()
                # Remove single-line comments (content after //)
                line = re.sub('//.*', '', line)
                code += line + '\n'
            # Remove multi-line comments (content between /* and */)
            code = re.sub('/\\*.*?\\*/', '', code)
            code = clean_gadget([code])
            nor_code.append(code[0])
        return nor_code
    
    def normalization_df(self, df: DataFrame):
        """Normalize a DataFrame containing source code."""
        df['code'] = df['code'].apply(lambda x: self.normalization({'code': [x]})[0])
        print(df['code'][0])  # Print the first normalized code for debugging
        return df   
    
if __name__ == '__main__':
    # Example usage
    gadget = [
        'int foo() {',
        '   // This is a comment',
        '   /* This is a multi-line comment */',
        '   struct ClassB obj;',
        '   int x = 1;',
        '   int y = 2;',
        '   return x + y;',
        '}'
    ]
    normalizer = CXXNormalization()
    cleaned = normalizer.normalization({'code': gadget})
    print(cleaned)