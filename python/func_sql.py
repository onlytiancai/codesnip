from itertools import groupby
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
            func_name, arg = re.match('(\w+)\((\w+)\)', item).groups()
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

data = [{'gender': 'boy', 'age': 18},
        {'gender': 'boy', 'age': 20},
        {'gender': 'girl', 'age': 16},
        {'gender': 'girl', 'age': 18},
        {'gender': 'boy', 'age': 56},
       ] 

query = select('avg(age), min(age), max(age)').from_(data).groupby('gender')
for item in query.run():
    print(item)
