# -*- coding: utf-8 -*-
import re
from pprint import pprint
from collections import namedtuple

rules = [
    ('(', r'\('),
    (')', r'\)'),
    ('{', r'\{'),
    ('}', r'\}'),
    ('+', r'\+'),
    ('-', r'\-'),
    ('*', r'\*'),
    ('/', r'\/'),
    ('%', r'\%'),
    ('=', r'='),
    ('<', r'<'),
    ('>', r'>'),
    ('==', r'=='),
    ('<=', r'<='),
    ('>=', r'>='),
    (';', r';'),
    ('if', r'if'),
    ('while', r'while'),
    ('num', r'\d+'),
    ('name', r'\w+'),
]


Rule = namedtuple('Rule', 'name pattern')
MatchResult = namedtuple('MatchResult', 'rule result')
rules = list(map(lambda x: Rule(x[0], x[1]), rules))


def parse_token(input):
    '根据正则规则做最长匹配，返回匹配结果及剩余字符'
    results = map(lambda rule: MatchResult(rule, re.match(rule.pattern, input)), rules)

    results = list(filter(lambda x: x.result is not None, results))
    if len(results) < 1:
        raise Exception('Unknow input: %s' % input.split()[0])

    max_result = max(results, key=lambda x: x.result.end())
    return max_result, input[max_result.result.end():].lstrip()


def parse_tokens(input):
    '把字符串解析成 token 数组'
    ret = []

    match, tail = parse_token(input)
    ret.append(match[1].group())
    while len(tail) > 0:
        match, tail = parse_token(tail)
        ret.append(match[1].group())
    return ret


def parse_ast(tokens):
    '''
    P -> S $

    S -> id = EXP S'
    S -> print ( EXPS ) S'
    S -> if ( EXP ) { S } else { S }
    S -> while ( EXP ) { S }
    S' -> ; S
    S' -> 

    EXPS -> EXP EXPS'
    EXPS' -> , EXPS
    EXPS' ->

    EXP -> ( EXP )
    EXP -> int EXP' 
    EXP -> id EXP'
    EXP' -> OP EXP
    EXP' ->

    OP -> +
    OP -> -
    OP -> *
    OP -> /
    '''
    pass


def run_ast(ast):
    pass


def eval(input):
    tokens = parse_tokens(input)
    ast = parse_ast(tokens)
    result = run_ast(ast)
    return input
