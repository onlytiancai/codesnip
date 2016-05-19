#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
DNSPod命令行版, 需要申请login_token，参考：https://support.dnspod.cn/Kb/showarticle/tsid/227/

帮助

    python dpcli.py --help
    python dpcli.py domain --help
    python dpcli.py record --help

示例：
    
    # 设置login_token环境变量
    export DP_LOGIN_TOKEN='10000,e31588236fe82510c7'
    # 查看域名列表
    python dpcli.py domain list
    # 查看mydomain.com记录列表
    python dpcli.py record list domain=mydomain.com
    # 给mydomain.com添加一条记录
    python dpcli.py record create domain=mydomain.com sub_domain=www record_type=A record_line=默认 value=8.8.8.8

'''
import os
import sys
import json
import string
import urllib
import urllib2
import argparse

BASE_URI = 'https://dnsapi.cn'
DP_LOGIN_TOKEN = os.environ.get('DP_LOGIN_TOKEN')

if not DP_LOGIN_TOKEN:
    print u'DP_LOGIN_TOKEN environment variable not found.'
    sys.exit()

cmd_groups = {
    'domain': ['create', 'list', 'remove', 'status', 'info'],
    'record': ['create', 'list', 'remove', 'modify', 'remark'],
}


def encode(obj):
    if not isinstance(obj, basestring):
        return str(obj)

    english = True
    for s in obj:
        if s not in string.printable:
            english = False

    return obj if english else repr(obj)


def print_result(rsp):
    for key in rsp:
        if isinstance(rsp[key], dict):
            print '=' * 20, key
            for key2 in rsp[key]:
                print '%s.%s: %s' % (key, key2, encode(rsp[key][key2]))
        elif isinstance(rsp[key], list):
            for i, obj in enumerate(rsp[key]):
                print '=' * 20, key, i
                for key2 in obj:
                    print '%s.%s: %s' % (key, key2, encode(obj[key2]))


def domain_api(*args):
    try:
        api = '%s/%s.%s' % (BASE_URI, args[0].capitalize(), args[1].capitalize())
        data = dict(arg.split('=') for arg in args[2:] if arg.find('=') > 0)
        print api, data
        data.update({"login_token": DP_LOGIN_TOKEN, "format": "json"})
        rsp = urllib2.urlopen(url=api, data=urllib.urlencode(data))
        print_result(json.loads(rsp.read()))
    except Exception, ex:
        raise
        print ex


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, 
        description=__doc__)
sub_group = parser.add_subparsers(title='subcommands')
for group_name in cmd_groups:
    group = sub_group.add_parser(group_name)
    sub_cmd = group.add_subparsers()
    for cmd in cmd_groups[group_name]:
        p = sub_cmd.add_parser(cmd)
        p.add_argument('args', type=str, nargs='*')


if __name__ == '__main__':
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()
    args = parser.parse_args()
    domain_api(*sys.argv[1:])
