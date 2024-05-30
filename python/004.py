from collections import namedtuple
import re

Token = namedtuple('Token', ['type', 'value'])

rules = [
    [r'\d+', 'N'],
    [r'\+', '+'],
    [r'-', '-'],
    [r'\*', '*'],
    [r'/', '/'],
    [r'\(', '('],
    [r'\)', ')'],
    [r'\w+', 'ID'],
    [r'\s+', 'IGNORE'],
]

def parse(s):
    while True:
        origin = s
        for patt, type in rules:
            m = re.match(patt, s)
            if m:
                if type != 'IGNORE':
                    yield Token(type, s[:m.end()])
                s = s[m.end():]
        if not s:
            break
        if origin == s:
            raise Exception('Unexpect token:', s[0])

for token in parse('(a + b) * 3 / 4##'):
    print(token)
