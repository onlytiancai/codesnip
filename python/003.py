from collections import namedtuple
from typing import List

tokens = ('NUMBER', 'PLUS', 'MINUS', 'TIMES', 'DIVIDE')
Token = namedtuple('Token', ['type', 'value'])
ASTNode = namedtuple('ASTNode', ['type', 'value', 'children'])

def parse(str: str) -> List[Token]:
    pass

def analyze(tokens: List[Token]) -> ASTNode:
    pass

def evaluate(ast: ASTNode) -> float:
    pass

def run(input: str):
    return evaluate(analyze(parse(input)))

print(run('2+3*4-5'))
