import re
from typing import NamedTuple, Iterable
from itertools import groupby
import logging
from datetime import datetime

logger = logging.getLogger('logsql')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

def format_time(time, format):
    if not isinstance(time, datetime):
        raise Exception('type error for time')
    m = re.match(r'^(\d+)([hms])$', format)
    if not m:
        raise Exception(f'unknown format:{format}')
    n, unit = m.groups()
    seconds = 1
    if unit == 'h':
        seconds = int(n) * 60 * 60
        time = time.replace(minute=0, second=0)
    elif unit == 'm':
        seconds = int(n) * 60
        time = time.replace(second=0)
    elif unit == 's':
        seconds = int(n)
        pass

    return datetime.fromtimestamp(int(time.timestamp()/seconds)*seconds)


class Token(NamedTuple):
    name: str
    enclosed: str 
    type: str
    format: str

supported_enclosed = ['""', "''", '[]']

def _get_value(token, token_value):
    if token.type == 'int':
        return int(token_value)
    elif token.type == 'float':
        return float(token_value)
    elif token.type == 'time':
        return datetime.strptime(token_value, token.format) 
    else:
        return token_value

def parse(rule, line):
    tokens = _getTokens(rule)
    ret = {}
    for token in tokens:
        regex = r'(\s+|$)'
        if token.enclosed:
            line=line[1:]
            regex = r'(?<!\\)'+token.enclosed[-1]+r'(\s+|$)?'

        m = re.search(regex, line)
        if m:
            if token.name != '-':
                token_value = line[m.pos:m.start()]
                ret[token.name] = _get_value(token, token_value) 
            line = line[m.end():]
    return ret 

def _parseToken(token_name, token_enclosed):
    # for `time:time:%Y-%m-%dT%H:%M:%S%z`
    m = re.match(r'(\w+):(\w+):([\w%:-]+)', token_name)
    if m:
        arr = m.groups()
        return Token(arr[0], token_enclosed, arr[1], arr[2])

    # for `status_code:int`
    m = re.match(r'(\w+):(\w+)', token_name)
    if m:
        arr = m.groups()
        return Token(arr[0], token_enclosed, arr[1], '')

    # for `name`
    return Token(token_name, token_enclosed, 'str', '')

def _parseEnclosed(token):
    for enclosed in supported_enclosed:
        if token[0] == enclosed[0]:
            if token[-1] != enclosed[-1]:
                raise Exception(f"error token:{token}")
            return token.strip(enclosed), enclosed
    return token, ''

def _getTokens(rule):
    tokens = []
    str_tokens = re.split(r'\s+', rule) 
    for token in str_tokens:
        token_name, token_enclosed = _parseEnclosed(token)
        tokens.append(_parseToken(token_name, token_enclosed))
        logger.debug('tokens:%s', tokens)
    return tokens

class AvgFun(object):
    total = 0
    len = 0
    def __init__(self, name, key=None):
        self.name = name
        self.key = key

    def hit(self, data):
        self.total += float(self.key(data) if self.key else data)
        self.len += 1

    def result(self):
        result = self.total/self.len
        self.total = 0
        self.len = 0
        return result

class MinFun(object):
    ret = float('inf') 
    def __init__(self, name, key=None):
        self.name = name
        self.key = key

    def hit(self, data):
        value = self.key(data) if self.key else data
        if value < self.ret:
            self.ret = value

    def result(self):
        result = self.ret
        self.ret = float('inf')
        return result

class MaxFun(object):
    ret = float('-inf') 
    def __init__(self, name, key=None):
        self.name = name
        self.key = key

    def hit(self, data):
        value = self.key(data) if self.key else data
        if value > self.ret:
            self.ret = value

    def result(self):
        result = self.ret
        self.ret = float('-inf')
        return result

funs = {}
funs['left'] = lambda s,l: s[:l]

def _split_select(txt):
    in_bracket = 0
    current = ''
    for ch in txt:
        current += ch
        if ch == '(':
            in_bracket += 1
        if ch == ')':
            in_bracket -= 1
        if ch == ',' and in_bracket == 0:
            yield current[:-1]
            current = ''
    if current:
        yield current

clas Query(object):
    def __init__(self):
        self.selected = [] 
        self.data = None
        self.group_name = None

    def select(self, selected):
        for item in _split_select(selected):
            matched = re.match(r'(\w+)\((\w+)\)', item)
            if matched:
                func_name, arg = matched.groups()
                if func_name == 'avg':
                    self.selected.append(AvgFun(item, lambda x: x[arg]))
                elif func_name == 'min':
                    self.selected.append(MinFun(item, lambda x: x[arg]))
                elif func_name == 'max':
                    self.selected.append(MaxFun(item, lambda x: x[arg]))
                else:
                    raise Exception(f'unknown function:{func_name}')
            else:
                self.selected.append(item)
        logger.debug('selected:%s', self.selected)
        return self

    def from_(self, data):
        self.data = data
        logger.debug('from:%s', self.data)
        return self

    def groupby(self, group_name):
        self.group_name = group_name
        logger.debug('group_name:%s', self.group_name)
        return self

    def _filtered_data(self, data):
        for line in data:
            if eval(self.conditions, funs, line):
                yield line

    def filter(self, conditions):
        if conditions:
            self.conditions = conditions
            logger.debug('conditions:%s', self.conditions)
            self.data = self._filtered_data(self.data) 
        return self

    def _group_key(self, x):
        if self.group_name in x:
            return x[self.group_name]
        else:
            # for `format_time(time,"1h")`
            m = re.match(r'^(\w+)\(.+\)$', self.group_name)
            if m:
                funcs = {
                    'format_time': format_time
                }
                return eval(self.group_name, funcs, x)

    def run(self):
        if self.group_name:
            for k, g in groupby(self.data, key=self._group_key):
                result = {}
                for item in g:
                    for x in self.selected:
                        if not isinstance(x, str):
                            x.hit(item)

                for x in self.selected:
                    if not isinstance(x, str):
                        result[x.name] = x.result()
                if self.group_name in self.selected:
                    result[self.group_name] = k
                yield result
        else:
            for line in self.data:
                result = dict((k,v) for k,v in line.items() if k in self.selected) 
                for x in self.selected:
                    if x not in line:
                        result[x] = eval(x, funs, line)
                yield result

def select(selected):
    return Query().select(selected)

def data_stream(rule, path):
    for line in open(path):
        result = parse(rule, line)
        logger.debug('parse result:%s', result)
        yield result 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(prog='logsql',description='快速分析日志')
    parser.add_argument('log_path', type=str, help='日志路径')
    parser.add_argument('rule', help='解析规则')
    parser.add_argument('-s', '--select', help='选择哪些列', default='*')
    parser.add_argument('-w', '--where', help='过滤条件', default='')
    parser.add_argument('-g', '--group', help='分组列', default='')
    parser.add_argument('-v', '--verbosity', help='显示调试信息', action='store_true')
    args = parser.parse_args()
    if args.verbosity:
        logger.setLevel(logging.DEBUG)

    stream = data_stream(args.rule, args.log_path) 
    query = select(args.select).from_(stream).filter(args.where).groupby(args.group)
    
    for line in query.run():
        print(line)
