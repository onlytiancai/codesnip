#encoding=utf-8
from fnmatch import fnmatch
from os import walk
from os.path import join,exists
import sys
import re

ignore_patterns = []
if exists('.ignore_patterns'):
    ignore_patterns = [re.compile(line.rstrip('\r\n')) 
            for line in open('.ignore_patterns')]
def ignore_line(line):
    '该行是否是忽略检查'
    for pattern in ignore_patterns:
        if pattern.match(line):
            return True
    return False

def get_args():
    '返回搜索路径，及文件匹配模式'
    arg_len = len(sys.argv)
    if arg_len == 1:
        return '.', '*'
    if arg_len == 2:
        return '.', sys.argv[1] 
    if arg_len == 3:
        return sys.argv[1], sys.argv[2]
    raise Exception('too many args')
 
def get_files(rootdir='.', match='*'):
    '返回需要遍历的文件'
    for root,dirs,files in walk(rootdir):
        for f in files:
            if fnmatch(f, match):
                yield join(root, f)

def calc_repeat_line(files):
    '返回重复的行的信息'
    result = {}
    
    def get_clear_line(line):
        '去掉空白后重组'
        return ''.join(line.split())
    
    def process_file(f):
        for line_no, line in enumerate(open(f)):
            if ignore_line(line):continue
            clear_line = get_clear_line(line)
            result.setdefault(clear_line, []) 
            result[clear_line].append((f, line_no, line))

    for f in files:
        process_file(f)
    return result

def enum_sort_result(result):
    items = sorted(result.items(), key=lambda d:len(d[1]))
    for key,value in items:
        yield result[key]

def get_result():
    rootdir, match = get_args()
    files = get_files(rootdir, match)
    result = calc_repeat_line(files)
    return enum_sort_result(result)

def print_result():
    for line_infos in get_result():
        repeat_len = len(line_infos)
        if repeat_len < 2: continue

        print line_infos[0][2].rstrip()
        print '*'*50
        for line_info in line_infos:
            print '\t', line_info[0], line_info[1]
        print '='*50
print_result()
