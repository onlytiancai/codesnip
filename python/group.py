#encoding=utf-8
from optparse import OptionParser
import re

def get_args():
    def get_parser():
        usage = u"""%prog -f filename -r rule [-d] [-c]
用途：对文本文件按照指定模式进行分组并排序,主要分析文本日志用
注意：如果正则表达式里有分组，则提取第一分组，
      如果不希望这样，请使用正则的无捕获分组(?:)
示例：统计日志里每分钟的日志量，默认按时间正序排列
      python group.py -f log.txt -r "\d\d\d\d\-\d\d\-\d\d \d\d:\d\d"
      统计日志里每个ip出现的次数，并按出现次数倒序排列
      python group.py -f input.txt -r "\d+\.\d+.\d+.\d+" -c -d"""
        return OptionParser(usage)

    def add_option(parser):
        parser.add_option("-f", "--file", dest="filename",
           help=u"需要分组的文本文件", metavar="FILE")
        parser.add_option("-r", "--rule", dest="rule",
           help=u"分组的正则表达式", metavar="REGEX")
        parser.add_option("-d", action="store_true", 
            dest="reverse", default=False, help=u"反序排列")
        parser.add_option("-c", action="store_true", 
            dest="orderbycount", default=False, 
            help=u"按数量排序,默认按匹配字符串排序")

    def get_options(parser):
        options, args = parser.parse_args()
        if not options.filename:
            parser.error('没有指定文件名')
        if not options.rule:
            parser.error('没有指定分组规则')
        return options

    parser = get_parser()
    add_option(parser)
    return get_options(parser)

options      = get_args()
filename     = options.filename
rule         = options.rule
reverse      = options.reverse
orderbycount = options.orderbycount 
regex        = re.compile(rule, re.IGNORECASE)
keys         = {}

def counter_key(key):
    keys.setdefault(key, 0)
    keys[key] += 1

def print_keys():
    sort_key = lambda d:d[1] if orderbycount else lambda d:d[0]
    temp_items = sorted(keys.items(), key=sort_key, reverse=reverse)
    for item in temp_items:
        key = item[0]
        print key, keys[key]

def get_key(line):
    m = regex.search(line)
    if m:
        return m.group() if regex.groups == 0 else m.group(1)
    return '!NotMatch!' 

with open(filename) as f:
    for line in f:
        key = get_key(line)
        counter_key(key)

print_keys()
