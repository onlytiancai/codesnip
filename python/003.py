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
            if ch in ('+', '-', '*', '/'):
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
    '''
    add -> mul (+ mul)* | mul (- mul)*
    mul -> pri (* pri)* | pri (/ pri)*
    pri -> Num
    '''
    global curr
    curr, tokens_count = 0, len(tokens)

    def peek():
        return tokens[curr] if curr < tokens_count else Token('', '') 

    def read():
        global curr
        ret = tokens[curr] 
        curr += 1
        return ret

    def pri():
        token = read()
        if token.type != 'N':
            raise Exception(f'expect numbers: {curr}:{token}')
        return ASTNode(token.type, token.value, [])

    def mul():
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

    def add():
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
    ret = add()

    print('analyze result:')
    def print_tree(node, level=0):
        print(level*'\t' + str(node.value))
        for child in node.children:
            print_tree(child, level+1)
    print_tree(ret)
    return ret

def evaluate(ast: ASTNode) -> float:
    pass

def run(input: str):
    return evaluate(analyze(parse(input)))

print(run('22+333*4-5'))
