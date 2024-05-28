from collections import namedtuple
from typing import List

Token = namedtuple('Token', ['type', 'value'])
ASTNode = namedtuple('ASTNode', ['type', 'value', 'children'])

def parse(str: str) -> List[Token]:
    tokens, curr = [], ''
    for ch in str:
        if '0' <= ch <='9':
            curr += ch
        else:
            if curr:
                tokens.append(Token('N', int(curr)))
                curr = ''
            if ch in ('+', '-', '*', '/', '(', ')'):
                tokens.append(Token(ch, ch))
            else:
                raise Exception(f'unexpect token:{ch}')
    if curr:
        tokens.append(Token('N', int(curr)))
        curr = ''

    print('parse result:')
    for token in tokens:
        print(token)

    return tokens 

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
            return ASTNode(token.type, token.value, [])
        elif token.type == '(':
            node = add()
            match(')')
            return node
        else:
            raise Exception(f'expect numbers: {token_index}:{token}')

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
