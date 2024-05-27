from collections import namedtuple
from typing import List

Token = namedtuple('Token', ['type', 'value'])
ASTNode = namedtuple('ASTNode', ['type', 'value', 'children'])

def parse(str: str) -> List[Token]:
    ret, curr = [], ''
    for ch in str:
        if '0' <= ch <='9':
            curr += ch
        else:
            if curr:
                ret.append(Token('N', int(curr)))
                curr = ''
            if ch in ('+', '-', '*', '/'):
                ret.append(Token(ch, ch))
            else:
                raise Exception(f'unexpect token:{ch}')
    print('parse:', ret)
    return ret

def analyze(tokens: List[Token]) -> ASTNode:
    pass

def evaluate(ast: ASTNode) -> float:
    pass

def run(input: str):
    return evaluate(analyze(parse(input)))

print(run('22+333*4-5'))
