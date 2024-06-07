'''
prog
    : stmt+ 
    ;

stmt
    : exp ';'
    | ID '=' exp ';' 
    | IF '(' exp ')' block (ELSE block )?
    | WHILE '(' exp ')' block
    | BREAK ';'
    | SEMI
    ;

block
    : '{' stmt+ '}'
    ;

exp
    : or 

or
    : and 
    | or || and

or
    : and or'
or'
    : || and or'
    | empty 

or
    : and (|| and)*

and
    : equal 
    | and && equal

and
    : equal and'
    | && equal and'
    | empty

and
    : equal (&& equal)*

equal
    : rel 
    | equal == rel 
    | equal != rel

equal
    : rel equal'

equal'
    : == rel equal'
    | != rel equal'
    | empty

equal
    : rel (== rel)*
    | rel (!= rel)*

rel 
    : add 
    | rel > add 
    | rel < add 
    | rel >= add 
    | rel <= add

rel
    : add rel'

rel'
    : > add rel'
    | < add rel'
    | >= add rel'
    | <= add rel'
    | empty

rel
    : add (> add)*
    | add (< add)*
    | add (>= add)*
    | add (<= add)*

add
    : mul 
    | add + mul 
    | add - mul

add
    : mul add'

add'
    : + mul add'
    | - mul add'
    | empty

add
    : mul (+ mul)*
    | mul (- mul)*

mul
    : pri 
    | mul * pri 
    | mul / pri

mul
    : pri mul'

mul'
    : * pri mul'
    | / pri mul'

mul
    : pri (* pri)*
    | pri (/ pri)*
    ;

pri
    : ID
    | funcCall 
    | NUM 
    | '(' exp ')' 
    | '-' pri
    ;

funcCall
    : ID '(' expList ')'

expList
    : exp (, exp)* 
    | empty
'''
from collections import namedtuple
from typing import List
import re

Token = namedtuple('Token', ['type', 'value'])
ASTNode = namedtuple('ASTNode', ['type', 'value', 'children'])

rules = [] 
for patt in ['\+', '-', '\*', '\/', '\(', '\)', ',', '=', 
             ';', '>', '<', '<=','>=','{', '}', 
             '\|\|', '&&', '==', '!=']:
    rules.append([patt, patt.replace('\\', '')])

rules.extend([
    [r'\d+', 'N'],
    [r'[a-zA-Z_]+', 'ID'],
    [r'\s+', 'IGNORE'],
    [r'"[^"]*"', 'S'],
])

print(rules)
def parse(s: str) -> List[Token]:
    ret = []
    while True:
        origin = s
        for patt, type in rules:
            m = re.match(patt, s)
            if m:
                if type != 'IGNORE':
                    value = s[:m.end()]
                    if type == 'ID' and value in ['if', 'else', 'while']:
                        type = value.upper()
                    ret.append(Token(type, value))
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
        if token_index >=  tokens_count:
            return Token('', '')
        ret = tokens[token_index] 
        token_index += 1
        return ret

    def unread():
        backto(token_index-1)

    def backto(index):
        global token_index 
        token_index = index 


    def match(token_type):
        token = read()
        if token.type != token_type:
            raise Exception('expect %s, got %s' % (token_type, token.type))

    def prog():
        '''
        prog
            : stmt+ 
            ;
        '''
        node = ASTNode('prog', 'prog', [])
        child = stmt()
        if not child:
            raise Exception('prog: at least one stmt is required')
        node.children.append(child)
        while True:
            print('prog aaaaaaaaaa:',child)
            child = stmt()
            print('prog bbbbbbbbb:',child)
            if not child:
                break
            node.children.append(child)

        return node

    def stmt():
        '''
        stmt
            : exp ';'
            | ID '=' exp ';' 
            | IF '(' exp ')' block (ELSE block )?
            | WHILE '(' exp ')' block
            | BREAK ';'
            | SEMI
            ;
        '''
        def expStmt():
            temp_index = token_index
            node = exp()
            print('expStmt', node)
            if node:
                token = peek()
                if token.value == ';':
                    read()
                    print('444', token)
                    return ASTNode('expStmt', 'expStmt', [node])
            backto(temp_index)

        def assignStmt():
            "| ID '=' exp ';'"
            temp_index = token_index
            token = peek()
            if token.type == 'ID':
                read()
                child = ASTNode('id', token.value, [])
                node = ASTNode('assignStmt', 'assignStmt', [child])
                token = peek()
                if token.type == '=':
                    read()
                    child = exp()
                    if not child:
                        raise Exception('assignStmt: exp required')
                    node.children.append(child)
                    match(';')
                    return node 
            backto(temp_index)

        def ifStmt(): 
            "IF '(' exp ')' block (ELSE block )?"
            temp_index = token_index
            token = peek()
            if token.type == 'IF':
                print('ifStmt:', 111)
                read()
                node = ASTNode('ifStmt', 'ifStmt', [])
                match('(')
                child = exp()
                if not child:
                    raise Exception('ifStmt: exp required')
                node.children.append(child)
                match(')')
                child = block()
                print('ifStmt:', 222)
                if not child:
                    raise Exception('ifStmt: if_block required')
                node.children.append(child)
                token = peek()
                if token.type == 'ELSE':
                    read()
                    child = block()
                    if not child:
                        raise Exception('ifStmt: else_block required')
                    node.children.append(child)
                return node

            backto(temp_index)

        def whileStmt(): 
            "WHILE '(' exp ')' block"
            temp_index = token_index
            token = peek()
            if token.type == 'WHILE':
                read()
                node = ASTNode('whileStmt', 'whileStmt', [])
                match('(')
                child = exp()
                if not child:
                    raise Exception('whileStmt: exp required')
                node.children.append(child)
                match(')')
                child = block()
                if not child:
                    raise Exception('ifStmt: else_block required')
                node.children.append(child)
                return node

            backto(temp_index)

        def breakStmt():
            "BREAK ';'"
            temp_index = token_index
            token = peek()
            if token.type == 'BREAK':
                read()
                node = ASTNode('breakStmt', 'breakStmt', [])
                match(';')
                return node
            backto(temp_index)

        def emptyStmt():
            "SEMI"
            temp_index = token_index
            token = peek()
            if token.type == ';':
                read()
                return ASTNode('emptyStmt', 'emptyStmt', [])
            backto(temp_index)

        def block():
            "'{' stmt+ '}'"
            temp_index = token_index
            token = peek()
            if token.type == '{':
                read()
                node = ASTNode('block', 'block', [])
                while True:
                    child = stmt(); 
                    print('block aaa', child)
                    if not child:
                        break
                    node.children.append(child)
                print('block bbb')
                match('}')
                return node
            backto(temp_index)

        for func in [expStmt, assignStmt, ifStmt, whileStmt, breakStmt, emptyStmt]:
            node = func()
            if node:
                print('000', func, node)
                return node


    def exp():
        '''
        exp
            : or 
        '''
        node = or_()
        return node

    def _bop(ops, child_func):
        node = child_func();
        if node:
            while True:
                token = peek()
                if token.value in ops:
                    read()
                    node = ASTNode(token.type, token.value, [node, child_func()])
                else:
                    break
        return node



    def or_():
        '''
        or
            : and 
            | or || and
        or
            : and or'
        or'
            : || and or'
            | empty 

        or
            : and (|| and)*
        '''
        return _bop(['||'], and_)

    def and_():
        '''
        and
            : equal 
            | and && equal

        and
            : equal and'
            | && equal and'
            | empty

        and
            : equal (&& equal)*
        '''
        return _bop(['&&'], equal)

    def equal():
        '''
        equal
            : rel equal'

        equal'
            : == rel equal'
            | != rel equal'
            | empty

        equal
            : rel (== rel)*
            | rel (!= rel)*
        '''
        return _bop(['==', '!='], rel)

    def rel():
        '''
        rel 
            : add 
            | rel > add 
            | rel < add 
            | rel >= add 
            | rel <= add

        rel
            : add rel'

        rel'
            : > add rel'
            | < add rel'
            | >= add rel'
            | <= add rel'
            | empty

        rel
            : add (> add)*
            | add (< add)*
            | add (>= add)*
            | add (<= add)*
        '''
        return _bop(['>', '<', '>=', '<='], add)

    def add():
        '''
        add
            : mul 
            | add + mul 
            | add - mul

        add
            : mul add'

        add'
            : + mul add'
            | - mul add'
            | empty

        add
            : mul (+ mul)*
            | mul (- mul)*
        '''
        return _bop(['+', '-'], mul)

    def mul():
        '''
        mul
            : pri mul'

        mul'
            : * pri mul'
            | / pri mul'

        mul
            : pri (* pri)*
            | pri (/ pri)*
            ;
        '''
        return _bop(['*','/'], pri)

    def pri():
        'pri -> -pri| Num | (add) | func_call | id | S'
        token = read()
        print('pri', token)
        if token.type == '-':
            return ASTNode('NEGATIVE', token.value, [pri()])
        if token.type == 'N':
            return ASTNode(token.type, float(token.value), [])
        elif token.type == '(':
            node = add()
            match(')')
            return node
        elif token.type == 'ID':
            next_token = peek()
            if next_token.type == '(':
                unread()
                return func_call()
            else:
                return ASTNode('id', token.value, [])
        elif token.type == 'S':
            return ASTNode(token.type, token.value.strip('"'), [])
        else:
            unread()

    def func_call():
        'func_call -> id ( expr_list)'
        token = read()
        print('3333', token, token_index)
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
        arg = exp()
        if arg:
            ret.append(arg)
            while True:
                token = read()
                if token.type == ',':
                    arg = exp()
                    ret.append(arg)
                else:
                    unread()
                    break
        return ret 

    ret = prog()
    print('analyze result:')
    def print_tree(node, level=0):
        print(level*'\t' + str(node.value))
        for child in node.children:
            print_tree(child, level+1)
    print_tree(ret)
    return ret

var_map = {}
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
    elif node.type == 'S':
        return node.value
    elif node.type == 'id':
        return var_map.get(node.value, 0)
    elif node.type == 'assignStmt':
        var_name = node.children[0].value
        val = evaluate(node.children[1])
        var_map[var_name] = val
    elif node.type == 'FUNC_CALL':
        if node.value == 'abs':
            return abs(evaluate(node.children[0]))
        elif node.value == 'pow':
            return pow(evaluate(node.children[0]), evaluate(node.children[1]))
        elif node.value == 'print':
            print('%s%s' % (node.children[0].value, evaluate(node.children[1])))
        else:
            raise Exception('Unexcept function name:%s' % node.value)
    elif node.type == 'prog':
        ret = None
        for child in node.children:
            ret = evaluate(child)
        return ret
    elif node.type == 'expStmt':
        return evaluate(node.children[0])
    elif node.type == 'block':
        for node in node.children:
            evaluate(node)
    elif node.type == 'ifStmt':
        cond = evaluate(node.children[0])
        if cond:
            evaluate(node.children[1])
        elif len(node.children) == 3:
            evaluate(node.children[2])
    else:
        raise Exception(f'unexpect node:{node}')

def run(input: str):
    ret = evaluate(analyze(parse(input)))
    print(var_map)
    return ret

expr = '2+3*4-5;'
ret = run(expr)
print(f'{expr} = {ret}')

expr = 'pow(abs(-2),4)+333*(4+-5)/2*abs(-6)-5-64+---5;'
ret = run(expr)
print(f'{expr} = {ret}')


expr = 'a=3;b=3;a+b;if(a-b){c=a*b;}else{c=a/b;}print("c=",c);c;'
ret = run(expr)
print(f'{expr} = {ret}')
