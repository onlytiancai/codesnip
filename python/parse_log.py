import re
from typing import NamedTuple, Iterable

class Token(NamedTuple):
    name: str
    enclosed: str 

supported_enclosed = ['""', "''", '[]']

def parse(rule, line):
    tokens = _getTokens(rule)
    ret = {}
    for token in tokens:
        # print(f'debug 1:[{line}] {token}')
        if not token.enclosed:
            m = re.search('(\s+|$)', line)
            if m:
                if token.name != '-':
                    ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
        else:
            line=line[1:]
            m = re.search(r'(?<!\\)'+token.enclosed[-1]+'(\s+|$)?', line)
            if m:
                if token.name != '-':
                    ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
    return ret 

def _getTokens(rule):
    tokens = []
    str_tokens = re.split('\s+', rule) 
    for token in str_tokens:
        t = Token(token, '')
        for enclosed in supported_enclosed:
            if token[0] == enclosed[0]:
                if token[-1] != enclosed[-1]:
                    raise Exception(f"error token:{token}")
                t = Token(token.strip(enclosed), enclosed)
        tokens.append(t)

    return tokens

def _select(select, data):
    result = ''
    for item in select:
        if item == '*':
            for k in sorted(data.keys()):
                result += data[k] + ' '
        elif (item in data):
            result += data[item] + ' '
    return result.rstrip()

def _group_select(select, buffer):
    print('debug', select, buffer)
    result = ''
    for item in select:
        if item == 'count(*)':
            result += str(len(buffer)) + ' '
        elif (item in buffer[0]):
            result += buffer[0][item] + ' '
    return result.rstrip()

def query(log:Iterable[str], rule:str, select:list=['*'], filter:dict={}, group:str=''):
    result = []
    group_buffer = [] 
    last_group = None
    for line in log:
        data = parse(rule, line)
        cond = [data[k] == v for k, v in filter.items()]
        if all(cond):
            if group:
                current_group = data[group] 
                if last_group != current_group:
                    if group_buffer:
                        result.append(_group_select(select, group_buffer))
                    group_buffer = []
                    last_group = current_group
                group_buffer.append(data)
            else:
                selected = _select(select, data)
                result.append(selected)

    if group_buffer:
        result.append(_group_select(select, group_buffer))

    return result 

import unittest
class StatTest(unittest.TestCase):
    log = '''10:11 111 222
10:12 333 444
10:14 333 666
10:13 555 666'''
    rule = 'a b c'
    def test_filter(self):
        actual = query(self.log.splitlines(), self.rule, ['*'], {'b': '333'}) 
        expected = ['10:12 333 444', '10:14 333 666']
        self.assertListEqual(actual, expected)

        actual = query(self.log.splitlines(), self.rule, ['*'], {'b': '333', 'c': '444'}) 
        expected = ['10:12 333 444']
        self.assertListEqual(actual, expected)

    def test_group_count(self):
        actual = query(log=self.log.splitlines(), 
                       rule=self.rule, 
                       select=['b','count(*)'],
                       group='b'
                       )
        expected = ['111 1', '333 2', '555 1']
        self.assertListEqual(actual, expected)

class ParseTest(unittest.TestCase):
    '''根据规则解析文本日志为 json，以便对日志进行按字段的过滤统计
    - 解析规则的多字段用空格隔开
    - 字段支持包裹字符以支持字段内有空格或引号的情况'''

    def test_error_token(self):
        with self.assertRaisesRegex(Exception, 'error token:'):
            parse('a "b c', '')

    def test_base(self):
        rule = 'a  b c'
        line = '111 222 333'
        expected = {'a':'111', 'b':'222', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_enclose(self):
        rule = '[a]  "b" c'
        line = '[1"11] "2 22" 333'
        expected = {'a':'1"11', 'b':'2 22', 'c': '333'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_escape(self):
        rule = 'a "b" [c]'
        line = r'111 "2\" 66 \"" [3\]33]'
        expected = {'a': '111', 'b':r'2\" 66 \"', 'c': r'3\]33'} 
        self.assertDictEqual(parse(rule, line), expected)

    def test_skip_field(self):
        rule = 'a - b'
        line = r'111 222 333'
        expected = {'a': '111', 'b':'333'} 
        self.assertDictEqual(parse(rule, line), expected)

if __name__ == '__main__':
    unittest.main()
