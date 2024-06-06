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
    | or = exp

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
for patt in ['\+', '-', '\*', '\/', '\(', '\)', ',', '=', ';', '>', '<','{', '}']:
    rules.append([patt, patt.replace('\\', '')])

rules.extend([
    [r'\d+', 'N'],
    [r'[a-zA-Z_]+', 'ID'],
    [r'\s+', 'IGNORE'],
])

def parse(s: str) -> List[Token]:
    ret = []
    while True:
        origin = s
        for patt, type in rules:
            m = re.match(patt, s)
            if m:
                if type != 'IGNORE':
                    value = s[:m.end()]
                    if type == 'ID' and value in ['if', 'while']:
                        type == value.upper()
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
                node = ASTNode('assignStmt', 'assignStmt', [token])
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
                read()
                node = ASTNode('ifStmt', 'ifStmt', [])
                match('(')
                child = exp()
                if not child:
                    raise Exception('ifStmt: exp required')
                node.children.append(child)
                match(')')
                child = block()
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
                while true:
                    child = exp()
                    if not child:
                        break
                    node.children.append(child)
                match('}')
                return node
            backto(temp_index)

        for func in [expStmt, assignStmt, ifStmt, whileStmt, breakStmt, emptyStmt]:
            node = func()
            if node:
                print('000', func, node)
                return node

        token = peek()
        if token.type != '':
            raise Exception('stmt: unexpect stmt type')



    def exp():
        'expr -> add'
        node = add()
        print('exp', node);
        return node

    def add():
        'add -> mul (+ mul)* | mul (- mul)*'
        child1 = mul();
        print('add', child1)
        node = child1
        if child1:
            while True: 
                token = peek()
                print('666', token)
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
        print('mul', child1)
        node = child1
        if child1:
            while True: 
                token = peek()
                print('555', token)
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
            token = peek()
            if token.type == '(':
                print('222', token, token_index)
                unread()
                return func_call()
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
    elif node.type == 'prog':
        ret = None
        for child in node.children:
            ret = evaluate(child)
        return ret
    elif node.type == 'expStmt':
        return evaluate(node.children[0])
    else:
        raise Exception(f'unexpect node:{node}')

def run(input: str):
    return evaluate(analyze(parse(input)))

expr = 'pow(abs(-2),4)+333*(4+-5)/2*abs(-6)-5-64+---5;'
ret = run(expr)
print(f'{expr} = {ret}')


expr = 'a=2;b=3;if(a+b>0){c=a+b};while(c<0){print(c);c=c-1;}'
ret = run(expr)
print(f'{expr} = {ret}')
