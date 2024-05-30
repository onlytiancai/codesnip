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
        if not s:
            break
        if origin == s:
            raise Exception('Unexpect token:', s[0])
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
        'pri -> Num | (add)'
        token = read()
        if token.type == 'N':
            return ASTNode(token.type, float(token.value), [])
        elif token.type == '(':
            node = add()
            match(')')
            return node
        else:
            raise Exception(f'expect numbers: {token_index}:{token}')

    def func_call():
        'func_call -> id ( expr_list)'
        pass

    def expr_list():
        'expr_list -> expr (expr,)*'
        pass

    ret = add()
    print('analyze result:')
    def print_tree(node, level=0):
        print(level*'\t' + str(node.value))
        for child in node.children:
            print_tree(child, level+1)
    print_tree(ret)
    return ret

def evaluate(node: ASTNode) -> float:
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
    else:
        raise Exception(f'unexpect node:{node}')

def run(input: str):
    return evaluate(analyze(parse(input)))

expr = '22+333*(4-5)/2'
ret = run(expr)
print(f'{expr} = {ret}')
