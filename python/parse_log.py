import re
from typing import NamedTuple

class Token(NamedTuple):
    name: str
    enclosed: str 

supported_enclosed = ['""', "''", '[]']

def parse(rule, line):
    tokens = _getTokens(rule)
    ret = {}
    for token in tokens:
        print(f'debug 1:[{line}] {token}')
        if not token.enclosed:
            m = re.search('(\s+|$)', line)
            if m:
                ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
        else:
            line=line[1:]
            m = re.search(token.enclosed[-1]+'(\s+|$)', line)
            if m:
                ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
    return ret 

def _getTokens(rule):
    tokens = []
    str_tokens = re.split('\s+', rule) 
    for token in str_tokens:
        t = Token(token, '')
        for enclosed in supported_enclosed:
            if token[0] == enclosed[0]:
                if token[-1] != enclosed[-1]:
                    raise Exception(f"error token:{token}")
                t = Token(token.strip(enclosed), enclosed)
        tokens.append(t)

    return tokens


import unittest
class MyTest(unittest.TestCase):
    def test_error_token(self):
        with self.assertRaisesRegex(Exception, 'error token:'):
            parse('a "b c', '')

    def test_base(self):
        rule = 'a  b c'
        line = '111 222 333'
        expected = {'a':'111', 'b':'222', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_enclose(self):
        rule = 'a  "b" c'
        line = '111 "2 22" 333'
        expected = {'a':'111', 'b':'2 22', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

if __name__ == '__main__':
    unittest.main()
