import re
import sys
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
    def __init__(self, name, key=None):
        self.total = 0
        self.len = 0
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
    def __init__(self, name, key=None):
        self.ret = float('inf')
        self.name = name
        self.key = key

    def hit(self, data):
        value = self.key(data) if self.key else data
        if value < self.ret:
            self.ret = value

    def result(self):
        ret = self.ret
        self.ret = float('inf')
        return ret

class MaxFun(object):
    def __init__(self, name, key=None):
        self.ret = float('-inf')
        self.name = name
        self.key = key

    def hit(self, data):
        value = self.key(data) if self.key else data
        if value > self.ret:
            self.ret = value

    def result(self):
        ret = self.ret
        self.ret = float('-inf')
        return ret

class CountFun(object):
    def __init__(self, name):
        self.ret = 0
        self.name = name

    def hit(self, data):
        self.ret += 1

    def result(self):
        ret = self.ret
        self.ret = 0
        return ret

regexp = lambda s,r: re.match(r, s)
funs = {}
funs['left'] = lambda s,l: s[:l]
funs['regexp'] = regexp

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

def key_func(arg):
    return lambda x: x[arg]

class Query(object):
    def __init__(self):
        self.selected = []
        self.data = None
        self.group_name = None

    def select(self, selected):
        for item in _split_select(selected):
            matched = re.match(r'(\w+)\((\w*)\)', item)
            if matched:
                func_name, arg = matched.groups()
                if func_name == 'avg':
                    self.selected.append(AvgFun(item, key_func(arg)))
                elif func_name == 'min':
                    self.selected.append(MinFun(item, key_func(arg)))
                elif func_name == 'max':
                    self.selected.append(MaxFun(item, key_func(arg)))
                elif func_name == 'count':
                    self.selected.append(CountFun(item))
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
                # print(111, self.group_name, x['time'], eval(self.group_name, funcs, x))
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

def data_stream(rule, path, filter):
    file = open(path) if path != '-' else sys.stdin
    for line in file:
        try:
            if not regexp(line, filter):
                continue
            result = parse(rule, line)
        except:
            logger.error('parse error:%s', line)
            raise
        logger.debug('parse result:%s', result)
        yield result

def get_out_line(line, format, selected):
    if format == 'json':
        return line
    arr = []
    for x in selected:
        k = x if isinstance(x, str) else x.name
        if k in line:
            arr.append(str(line[k]))
    return '\t'.join(arr)

if __name__ == '__main__':
    import io
    import os
    import time

    try:
        # open stdout in binary mode, then wrap it in a TextIOWrapper and enable write_through
        sys.stdout = io.TextIOWrapper(open(sys.stdout.fileno(), 'wb', 0), write_through=True)
        # for flushing on newlines only use :
        # sys.stdout.reconfigure(line_buffering=True)
    except TypeError:
        # In case you are on Python 2
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

    import argparse
    parser = argparse.ArgumentParser(prog='logsql',description='快速分析日志')
    parser.add_argument('log_path', type=str, help='日志路径')
    parser.add_argument('rule', help='解析规则')
    parser.add_argument('-s', '--select', help='选择哪些列', default='*')
    parser.add_argument('-w', '--where', help='过滤条件', default='')
    parser.add_argument('-g', '--group', help='分组列', default='')
    parser.add_argument('-v', '--verbosity', help='显示调试信息', action='store_true')
    parser.add_argument('-f', '--filter', help='过滤原始日志行', default='')
    parser.add_argument('-of', '--out_format', help='输出格式', default='json')
    args = parser.parse_args()
    if args.verbosity:
        logger.setLevel(logging.DEBUG)

    stream = data_stream(args.rule, args.log_path, args.filter)
    query = select(args.select).from_(stream).filter(args.where).groupby(args.group)

    for line in query.run():
        out_line = get_out_line(line, args.out_format, query.selected)
        try:
            print(out_line)
        except BrokenPipeError:
            break
