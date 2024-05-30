from collections import namedtuple
from typing import List
import re

Token = namedtuple('Token', ['type', 'value'])
ASTNode = namedtuple('ASTNode', ['type', 'value', 'children'])

rules = [
    [r'\d+', 'N'],
    [r'\+', '+'],
    [r'-', '-'],
    [r'\*', '*'],
    [r'/', '/'],
    [r'\(', '('],
    [r'\)', ')'],
    [r',', ','],
    [r'[a-zA-Z_]+', 'ID'],
    [r'\s+', 'IGNORE'],
]

def parse(s: str) -> List[Token]:
    ret = []
    while True:
        origin = s
        for patt, type in rules:
            m = re.match(patt, s)
            if m:
                if type != 'IGNORE':
                    ret.append(Token(type, s[:m.end()]))
                s = s[m.end():]
                break
        if not s:
            break
        if origin == s:
            raise Exception('Unexpect token:', s[0])
    for token in ret:
        print(token)
    return ret

def analyze(tokens: List[Token]) -> ASTNode:
    global token_index 
    token_index, tokens_count = 0, len(tokens)

    def peek():
        return tokens[token_index] if token_index < tokens_count else Token('', '') 

    def read():
        global token_index 
        ret = tokens[token_index] 
        token_index += 1
        return ret

    def unread():
        global token_index 
        token_index -= 1

    def match(token_type):
        token = read()
        if token.type != token_type:
            raise Exception(f'expect {token}, got {token_type}')

    def expr():
        'expr -> add'
        node = add()
        return node

    def add():
        'add -> mul (+ mul)* | mul (- mul)*'
        child1 = mul();
        node = child1
        if child1:
            while True: 
                token = peek()
                if token.value in ['+', '-']:
                    read()
                    child2 = mul()
                    node = ASTNode(token.type, token.value, [child1, child2])
                    child1 = node
                else:
                    break
        return node


    def mul():
        'mul -> pri (* pri)* | pri (/ pri)*'
        child1 = pri();
        node = child1
        if child1:
            while True: 
                token = peek()
                if token.value in ['*', '/']:
                    read()
                    child2 = pri()
                    node = ASTNode(token.type, token.value, [child1, child2])
                    child1 = node
                else:
                    break
        return node

    def pri():
        'pri -> -pri| Num | (add) | func_call'
        token = read()
        if token.type == '-':
            return ASTNode('NEGATIVE', token.value, [pri()])
        if token.type == 'N':
            return ASTNode(token.type, float(token.value), [])
        elif token.type == '(':
            node = add()
            match(')')
            return node
        elif token.type == 'ID':
            unread()
            return func_call()
        else:
            raise Exception(f'expect numbers: {token_index}:{token}')

    def func_call():
        'func_call -> id ( expr_list)'
        token = read()
        if token.type != 'ID':
            return None
        node = ASTNode('FUNC_CALL', token.value, [])
        match('(')
        for arg in args():
            node.children.append(arg)
        match(')')
        return node

    def args():
        'args -> expr (,expr)* | empty'
        ret = []
        arg = expr()
        if arg:
            ret.append(arg)
            while True:
                token = read()
                if token.type == ',':
                    arg = expr()
                    ret.append(arg)
                else:
                    unread()
                    break
        return ret 

    ret = expr()
    print('analyze result:')
    def print_tree(node, level=0):
        print(level*'\t' + str(node.value))
        for child in node.children:
            print_tree(child, level+1)
    print_tree(ret)
    return ret

def evaluate(node: ASTNode) -> float:
    if node.type == 'NEGATIVE':
        return -(evaluate(node.children[0]))
    if node.value == '+':
        return evaluate(node.children[0]) + evaluate(node.children[1])
    elif node.value == '-':
        return evaluate(node.children[0]) - evaluate(node.children[1])
    elif node.value == '*':
        return evaluate(node.children[0]) * evaluate(node.children[1])
    elif node.value == '/':
        return evaluate(node.children[0]) / evaluate(node.children[1])
    elif node.type == 'N':
        return node.value
    elif node.type == 'FUNC_CALL':
        if node.value == 'abs':
            return abs(evaluate(node.children[0]))
        elif node.value == 'pow':
            return pow(evaluate(node.children[0]), evaluate(node.children[1]))
        else:
            raise Exception('Unexcept function name:%s' % node.value)
    else:
        raise Exception(f'unexpect node:{node}')

def run(input: str):
    return evaluate(analyze(parse(input)))

expr = 'pow(abs(-2),4)+333*(4+-5)/2*abs(-6)-5-64+---5'
ret = run(expr)
print(f'{expr} = {ret}')
