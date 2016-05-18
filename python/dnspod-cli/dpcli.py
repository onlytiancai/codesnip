#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

DNSPod命令行版

帮助：

    ./dpcli.py --help
    ./dpcli.py domain --help
    ./dpcli.py record --help

示例：

    ./dpcli.py record create domain=ihuhao.com sub_domain=www record_type=A record_line=默认 value=8.8.8.8

'''
import sys
import json
import argparse

cmd_groups = {
    'domain': ['create', 'list', 'remove', 'status', 'info'],
    'record': ['create', 'list', 'remove', 'modify', 'remark'],
} 

def domain_api(*args):
    api = '%s.%s' % (args[0].capitalize(), args[1].capitalize())
    data = dict(arg.split('=') for arg in args[2:]) 
    print api, data 


parser = argparse.ArgumentParser()
sub_group = parser.add_subparsers(title='子命令')
for group_name in cmd_groups:
    group = sub_group.add_parser(group_name)
    sub_cmd = group.add_subparsers()
    for cmd in cmd_groups[group_name]:
        p = sub_cmd.add_parser(cmd)
        p.add_argument('args', type=str, nargs='+')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    domain_api(*sys.argv[1:])
