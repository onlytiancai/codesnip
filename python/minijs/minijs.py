# -*- coding: utf-8 -*-
import re
from pprint import pprint

rules = [
    ('(', r'\('),
    (')', r'\)'),
    ('+', r'\+'),
    ('-', r'\-'),
    ('*', r'\*'),
    ('/', r'\/'),
    ('=', r'='),
    (';', r';'),
    ('num', r'\d+'),
    ('var', r'var'),
    ('name', r'\w+'),
]


def get_patten(x): return x[1]


def parse_token(input):
    '根据正则规则做最长匹配，返回匹配结果及剩余字符'
    matchs = map(lambda rule: (rule, re.match(get_patten(rule), input)), rules)

    matchs = list(filter(lambda x: x[1] is not None, matchs))
    if len(matchs) < 1:
        raise Exception('Unknow input: %s' % input.split()[0])
    
    max_match = max(matchs, key=lambda x: x[1].end())
    return max_match, input[max_match[1].end():].lstrip()


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
    pass


def run_ast(ast):
    pass


def eval(input):
    tokens = parse_tokens(input)
    ast = parse_ast(tokens)
    result = run_ast(ast)
    return input
