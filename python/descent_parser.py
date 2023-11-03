#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
https://python3-cookbook.readthedocs.io/zh-cn/latest/c02/p19_writing_recursive_descent_parser.html

Topic: 下降解析器

    expr        -> term expr_tail 
    expr_tail   -> + term expr_tail
                 | - term expr_tail
                 | e

    term        -> factor term_tail
    term_tail   -> * factor term_tail
                 | / factor term_tail
                 | e

    factor      -> ( expr )
                 | NUM
                 | ALPHA alpha_tail
    alpha_tail  -> ( args ) # func call
                 | e        # var
    args        -> expr args_tail 
    args_tail   -> , expr
                 | e
"""
import re
import collections

# Token specification
NUM = r'(?P<NUM>\d+)'
PLUS = r'(?P<PLUS>\+)'
MINUS = r'(?P<MINUS>-)'
TIMES = r'(?P<TIMES>\*)'
DIVIDE = r'(?P<DIVIDE>/)'
LPAREN = r'(?P<LPAREN>\()'
RPAREN = r'(?P<RPAREN>\))'
WS = r'(?P<WS>\s+)'
ALPHA = r'(?P<ALPHA>\w+)'
COMMA = r'(?P<COMMA>,)'

master_pat = re.compile('|'.join([NUM, PLUS, MINUS, TIMES,
                                  DIVIDE, LPAREN, RPAREN, WS, ALPHA, COMMA]))
# Tokenizer
Token = collections.namedtuple('Token', ['type', 'value'])

def generate_tokens(text):
    scanner = master_pat.scanner(text)
    for m in iter(scanner.match, None):
        tok = Token(m.lastgroup, m.group())
        if tok.type != 'WS':
            yield tok

var_map = {'a': 10}
# Parser
class ExpressionEvaluator:
    '''
    Implementation of a recursive descent parser. Each method
    implements a single grammar rule. Use the ._accept() method
    to test and accept the current lookahead token. Use the ._expect()
    method to exactly match and discard the next token on on the input
    (or raise a SyntaxError if it doesn't match).
    '''

    def parse(self, text):
        self.tokens = generate_tokens(text)
        self.tok = None  # Last symbol consumed
        self.nexttok = None  # Next symbol tokenized
        self._advance()  # Load first lookahead token
        return self.expr()

    def _advance(self):
        'Advance one token ahead'
        self.tok, self.nexttok = self.nexttok, next(self.tokens, None)

    def _accept(self, toktype):
        'Test and consume the next token if it matches toktype'
        if self.nexttok and self.nexttok.type == toktype:
            self._advance()
            return True
        else:
            return False

    def _expect(self, toktype):
        'Consume next token if it matches toktype or raise SyntaxError'
        if not self._accept(toktype):
            raise SyntaxError('Expected ' + toktype)

    # Grammar rules follow
    def expr(self):
        'expr        -> term expr_tail'
        exprval = self.term()
        exprval = self.expr_tail(exprval)
        return exprval

    def expr_tail(self, left):
        '''
        expr_tail   -> + term expr_tail
                     | - term expr_tail
                     | e
        '''
        exprval = left
        if self._accept('PLUS'):
            exprval = left + self.term()
            exprval = self.expr_tail(exprval)
        if self._accept('MINUS'):
            exprval = left - self.term()
            exprval = self.expr_tail(exprval)
        return exprval

    def term(self):
        'term        -> factor term_tail'
        termval = self.factor()
        termval = self.term_tail(termval)
        return termval 

    def term_tail(self, left):
        '''
        term_tail   -> * factor term_tail
                     | / factor term_tail
                     | e
        '''
        termval = left
        if self._accept('TIMES'):
            termval = left * self.factor()
            termval = self.term_tail(termval)
        if self._accept('DIVIDE'):
            termval = left / self.factor()
            termval = self.term_tail(termval)
        return termval

    def factor(self):
        '''
        factor      -> ( expr )
                     | NUM
                     | ALPHA alpha_tail

        '''
        if self._accept('NUM'):
            return int(self.tok.value)
        elif self._accept('LPAREN'):
            exprval = self.expr()
            self._expect('RPAREN')
            return exprval
        elif self._accept('ALPHA'):
            return self.alpha_tail(self.tok.value)
        else:
            raise SyntaxError('Expected NUMBER or LPAREN')

    def alpha_tail(self, name):
        '''
        alpha_tail  -> ( args ) # func call
                     | e        # var
        '''
        if self._accept('LPAREN'):
            return self.func(name)
        return self.varval(name)

    def varval(self, name):
        if name in var_map:
            return int(var_map[name])
        raise Exception(f'unknown var name:{name}')

    def func(self, func_name):
        args = self.args()
        self._expect('RPAREN')
        if func_name == 'pow':
            return pow(*args)
        else:
            raise Exception(f'unknown function:{func_name}')

    def args(self):
        "args :: expr {',' expr}*"
        ret = []
        ret.append(self.expr())
        while self._accept('COMMA'):
            ret.append(self.expr())
        return ret

def descent_parser():
    e = ExpressionEvaluator()
    print(e.parse('2'))
    print(e.parse('2 + 3'))
    print(e.parse('2 + 3 * 4'))
    print(e.parse('2 + (3 + 4) * 5'))
    print(e.parse('a + 2 + pow(2,pow(2,2)) * 5'))
    # print(e.parse('2 + (3 + * 4)'))
    # Traceback (most recent call last):
    #    File "<stdin>", line 1, in <module>
    #    File "exprparse.py", line 40, in parse
    #    return self.expr()
    #    File "exprparse.py", line 67, in expr
    #    right = self.term()
    #    File "exprparse.py", line 77, in term
    #    termval = self.factor()
    #    File "exprparse.py", line 93, in factor
    #    exprval = self.expr()
    #    File "exprparse.py", line 67, in expr
    #    right = self.term()
    #    File "exprparse.py", line 77, in term
    #    termval = self.factor()
    #    File "exprparse.py", line 97, in factor
    #    raise SyntaxError("Expected NUMBER or LPAREN")
    #    SyntaxError: Expected NUMBER or LPAREN


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        descent_parser()
    elif len(sys.argv) == 2:
        e = ExpressionEvaluator()
        print(e.parse(sys.argv[1]))
    elif len(sys.argv) == 3:
        e = ExpressionEvaluator()
        var_map = dict(x.split('=') for x in sys.argv[1].split(',')) 
        expr = sys.argv[2]
        print(e.parse(expr))
