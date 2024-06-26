def parse(s):
    'parse words, numbers and math operators'
    ret = []
    temp = ''
    in_number = False
    in_alpha = False
    for ch in s:
        if '0' <= ch <= '9':
            if not in_number and temp != '':
                ret.append(temp)
                temp = ''
            in_alpha = False 
            in_number = True
            temp += ch
        elif 'a' <= ch <= 'z':
            if not in_alpha and temp != '':
                ret.append(temp)
                temp = ''
            in_number = False 
            in_alpha = True
            temp += ch
        elif ch in ['+', '-', '*', '/', '=', ';', ' ']:
            if temp != '':
                ret.append(temp)
                temp = ''
            in_alpha = False 
            in_number = False 
            if ch != ' ':
                ret.append(ch)
        else:
            raise Exception(f'unknown char:{ch}')
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

    def prog():
        'prog -> (stmt ;)* | add | empty'
        children = []
        while True:
            child = stmt()
            if not child:
                break
            children.append(child)
            if peek() == ';':
                next()
        if len(children) == 0:
            return None
        elif len(children) == 1:
            return children[0]
        else:
            return ['prog', children] 

    def stmt():
        'stmt -> add'
        return add()

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
        node = pri()
        if not node:
            return node
        while peek() == '*':
            next()
            node = ['mul', [node]]
            right = pri()
            if right is None:
                raise Exception('mul: right is none')
            node[1].append(right)
        return node
    
    def pri():
        'pri -> alpha | num'
        t = peek()
        if not t:
            return None
        if t.isalpha():
            return alpha()
        elif t.isdigit():
            return num()

    def alpha():
        ret = peek()
        if ret and ret.isalpha():
            next()
            return ret

    def num():
        ret = peek()
        if ret and ret.isdigit():
            next()
            return ret

    return prog() 

import unittest
class MyTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(['1','+','2','*','3'], parse('1+2*3'))

    def test_parse2(self):
        self.assertEqual(['1','+','2','*','3'], parse('1 + 2 * 3'))

    def test_parse3(self):
        self.assertEqual(['11','+','22','*','33'], parse('11 + 22 * 33'))

    def test_parse4(self):
        self.assertEqual(['a'], parse('a'))

    def test_parse5(self):
        self.assertEqual(['aa'], parse('aa'))

    def test_parse6(self):
        self.assertEqual(['aa','1'], parse('aa1'))

    def test_parse7(self):
        self.assertEqual(['aa','+', '12'], parse('aa+12'))

    def test_analyze(self):
        self.assertEqual(['add', [['mul', ['1', '2']], ['mul', ['3', '4']]]], analyze(parse('1*2+3*4')))

    def test_analyze2(self):
        with self.assertRaisesRegex(Exception, 'right is none'):
            analyze(parse('1*2+3*'))

    def test_analyze3(self):
        self.assertEqual('1', analyze(parse('1')))

    def test_analyze4(self):
        self.assertEqual(['add', ['1', '2']], analyze(parse('1+2')))

    def test_analyze5(self):
        self.assertEqual(None, analyze(parse('')))

    def test_analyze6(self):
        self.assertEqual('a', analyze(parse('a')))

    def test_analyze7(self):
        self.assertEqual('1', analyze(parse('1')))

    def test_analyze8(self):
        self.assertEqual(['prog', [ '1', 'a','2' ]], analyze(parse('1;a;2')))

if __name__ == "__main__":
    unittest.main(verbosity=2)
