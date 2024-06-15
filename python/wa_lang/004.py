import unittest
from pprint import pprint

def parse(s):
    ret = []
    temp = ''
    in_number = False
    for ch in s:
        if '0' <= ch <= '9':
            temp += ch
            if not in_number:
                in_number = True
        else:
            in_number = False 
            if temp != '':
                ret.append(temp)
                temp = ''
            if ch != ' ':
                ret.append(ch)
    if temp != '':
        ret.append(temp)
    return ret

def analyze(tokens):
    global token_index
    token_index = 0
    token_len = len(tokens)

    def next():
        global token_index
        token_index += 1

    def peek():
        return tokens[token_index] if token_index < token_len else None

    def read():
        ret = peek() 
        next()
        return ret

    def add():
        'add -> mul (+ mul)*'
        node = mul()
        if not node:
            return node
        while peek() == '+':
            next()
            node = ['add', [node]]
            right = mul()
            if right is None:
                raise Exception('add: right is none')
            node[1].append(right)
        return node

    def mul():
        'mul -> mul (* mul)*'
        node = num()
        if not node:
            return node
        while peek() == '*':
            next()
            node = ['mul', [node]]
            right = num()
            if right is None:
                raise Exception('mul: right is none')
            node[1].append(right)
        return node

    def num():
        ret = peek()
        if ret and ret.isdigit():
            next()
            return ret

    return add() 

class MyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(['1','+','2','*','3'], parse('1+2*3'))

    def test_parse2(self):
        self.assertEqual(['1','+','2','*','3'], parse('1 + 2 * 3'))

    def test_parse3(self):
        self.assertEqual(['11','+','22','*','33'], parse('11 + 22 * 33'))

    def test_analyze(self):
        self.assertEqual(['add', [['mul', ['1', '2']], ['mul', ['3', '4']]]], analyze(parse('1*2+3*4')))

    def test_analyze2(self):
        with self.assertRaises(Exception):
            analyze(parse('1*2+3*'))

if __name__ == "__main__":
    unittest.main(verbosity=2)
