import re
from typing import NamedTuple, Iterable
from itertools import groupby
import logging

logger = logging.getLogger('logsql')
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
logger.addHandler(ch)

class Token(NamedTuple):
    name: str
    enclosed: str 

supported_enclosed = ['""', "''", '[]']

def parse(rule, line):
    tokens = _getTokens(rule)
    ret = {}
    for token in tokens:
        if not token.enclosed:
            m = re.search(r'(\s+|$)', line)
            if m:
                if token.name != '-':
                    ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
        else:
            line=line[1:]
            m = re.search(r'(?<!\\)'+token.enclosed[-1]+r'(\s+|$)?', line)
            if m:
                if token.name != '-':
                    ret[token.name] = line[m.pos:m.start()]
                line = line[m.end():]
    return ret 

def _getTokens(rule):
    tokens = []
    str_tokens = re.split(r'\s+', rule) 
    for token in str_tokens:
        t = Token(token, '')
        for enclosed in supported_enclosed:
            if token[0] == enclosed[0]:
                if token[-1] != enclosed[-1]:
                    raise Exception(f"error token:{token}")
                t = Token(token.strip(enclosed), enclosed)
        tokens.append(t)

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

class Query(object):
    def __init__(self):
        self.selected = [] 
        self.data = None
        self.group_name = None

    def select(self, selected):
        for item in re.split(',\s*', selected):
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

    def run(self):
        if self.group_name:
            for k, g in groupby(self.data, key=lambda x: x[self.group_name]):
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
