import re
import pandas as pd
from clean_gadget import clean_gadget

def normalization(source):
    nor_code = []
    for fun in source['code']:
        lines = fun.split('\n')
        # print(lines)
        code = ''
        for line in lines:
            line = line.strip()
            # Remove single-line comments (content after //)
            line = re.sub('//.*', '', line)
            code += line + ' '
        # Remove multi-line comments (content between /* and */)
        # code = re.sub('(?<!:)\\/\\/.*|\\/\\*(\\s|.)*?\\*\\/', "", code)
        code = re.sub('/\\*.*?\\*/', '', code)
        code = clean_gadget([code])
        nor_code.append(code[0])
        print(code[0])
    return nor_code

def mutrvd():
    train = pd.read_pickle('trvd_train.pkl')
    test = pd.read_pickle('trvd_test.pkl')
    val = pd.read_pickle('trvd_val.pkl')

    train['code'] = normalization(train)
    train.to_pickle('./mutrvd/train.pkl')

    test['code'] = normalization(test)
    test.to_pickle('./mutrvd/test.pkl')

    val['code'] = normalization(val)
    val.to_pickle('./mutrvd/val.pkl')

if __name__ == '__main__':
    # mutrvd()
    gadget = [
        'int foo() {',
        '   // This is a comment',
        '   /* This is a multi-line comment */',
        '   int x = 1;',
        '   int y = 2;',
        '   return x + y;',
        '}'
    ]
    cleaned = clean_gadget(gadget)
    # concat cleaned code into a single string
    cleaned = ' '.join(cleaned)
    print(cleaned)
