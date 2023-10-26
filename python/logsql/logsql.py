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
    logger.debug('_group_select:%s %s', select, buffer)
    result = ''
    for item in select:
        if item == 'count(*)':
            result += str(len(buffer)) + ' '
        elif (item in buffer[0]):
            result += buffer[0][item] + ' '
    return result.rstrip()

def query(log:Iterable[str], rule:str, select:list=['*'], filter:dict={}, group:str=''):
    logger.debug('query begin: log=%s, rule=%s, select=%s, filter=%s, group=%s',
                  log, rule, select, filter, group)
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
import re

class AvgFun(object):
    total = 0
    len = 0
    def __init__(self, name, key=None):
        self.name = name
        self.key = key

    def hit(self, data):
        self.total += self.key(data) if self.key else data
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

class Query(object):
    selected = [] 
    data = None
    group_name = None

    def select(self, selected):
        for item in selected.split(', '):
            func_name, arg = re.match(r'(\w+)\((\w+)\)', item).groups()
            if func_name == 'avg':
                self.selected.append(AvgFun(item, lambda x: x[arg]))
            elif func_name == 'min':
                self.selected.append(MinFun(item, lambda x: x[arg]))
            elif func_name == 'max':
                self.selected.append(MaxFun(item, lambda x: x[arg]))
            else:
                raise Exception(f'unknown function:{func_name}')
        return self

    def from_(self, data):
        self.data = data
        return self

    def groupby(self, group_name):
        self.group_name = group_name
        return self

    def run(self):
        for k, g in groupby(self.data, key=lambda x: x[self.group_name]):
            result = {self.group_name: k}
            for item in g:
                for x in self.selected:
                    x.hit(item)

            for x in self.selected:
                result[x.name] = x.result()
            yield result

def select(selected):
    return Query().select(selected)


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

    # 'a=1 b=2 c:3' => {'a': '1', 'b': '2', 'c': '3'}
    where = dict([x.split('=') for x in args.where.split(' ') if x.find('=') >= 0])
    result = query(open(args.log_path), 
                   args.rule, 
                   args.select.split(' '),
                   where,
                   args.group,
             )
    for line in result:
        print(line)
