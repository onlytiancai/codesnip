import ply.lex as lex
import ply.yacc as yacc

tokens = [
    'LPAREN', 'RPAREN', 'COMMA', 'IDENTIFIER', 'NUMBER',
]

def lex_tokenize(text):
    lexer = lex.lexmodule('lex', {'literals': tokens})
    lexer.input(text)
    for tok in lexer:
        yield tok

def lex_test(text):
    lexer = lex.lexmodule('lex', {'literals': tokens})
    lexer.input(text)
    while True:
        tok = lexer.token()
        if not tok:
            break
        print(tok)

def parse_expr(p):
    parser = yacc.yacc(debug=True)
    return parser.parse(p)

# BNF 语法规则
def p_expr(p):
    'expr : LPAREN callee args RPAREN'
    return p[2](p[3])

def p_callee(p):
    'callee : IDENTIFIER'
    return p[1]

def p_args(p):
    '''args : empty 
            | expr COMMA args'''
    if len(p) > 2:
        return [p[1]] + p[3]
    else:
        return []

def p_empty(p):
    'empty : '
    pass

# 示例代码
text = "a(1, b(2, c(3)))"
#lex_test(text)
ast = parse_expr(text)
print(ast)
