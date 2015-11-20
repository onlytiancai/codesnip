#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
https://www.dnspod.cn/User
export dnspod_logintoken='10000, xxxx'

'''
import os
import json
import time
import socket
import urllib
import urllib2
import logging


logging.basicConfig(level=logging.NOTSET)

API_BASEURL = 'https://dnsapi.cn'
API_LOGIN_TOKEN = os.environ.get('dnspod_logintoken')
API_HEADERS = {"Content-type": "application/x-www-form-urlencoded",
               "Accept": "text/json",
               "User-Agent": "onlytiancai/0.0.1 (onlytiancai@gmail.com)"}

all_tasks = set()
error_task = set() 


def api(api, **data):
    url = '%s/%s' % (API_BASEURL, api)
    data.update(login_token=API_LOGIN_TOKEN, format='json')
    data = urllib.urlencode(data)
    req = urllib2.Request(url, data, headers=API_HEADERS)
    rsp = urllib2.urlopen(req)
    rsp = json.loads(rsp.read())
    if rsp['status']['code'] != '1':
        raise Exception(rsp['status']['message'])
    return rsp


def domain_list():
    ret = api('Domain.List')
    for item in ret['domains']:
        if item['status'] != 'enable' or item['ext_status'] != '':
            continue
        yield item['id'], item['name']


def record_list(domain_id):
    ret = api('Record.List', domain_id=domain_id)
    for item in ret['records']:
        if item['type'] not in ('A', 'CNAME'):
            continue
        if item['enabled'] != '1':
            continue
        yield item


def httptest(host, ip):
    url = 'http://%s/' % ip
    headers = {"Host": host}
    req = urllib2.Request(url, headers=headers)
    try:
        urllib2.urlopen(req, timeout=1)
    except urllib2.HTTPError, e:
        if e.code == 508:
            return '508 Loop Detected'
        return '%s %s' % (e.code, e.reason)
    except urllib2.URLError as e:
        if type(e.reason) is socket.timeout:
            return 'timeout'
        if type(e.reason) is socket.gaierror:
            return e.reason.strerror
        return e.reason
    except socket.timeout as e:
        return 'timeout'
    except socket.error as e:
        return e.strerror
    return ''


def get_host(domain, sub_domain):
    if sub_domain == '@':
        return domain
    if sub_domain == '*':
        sub_domain = 'test'
    return '%s.%s' % (sub_domain, domain)


def test_one(host, ip):
    if (host, ip) in all_tasks:
        return

    t1 = time.time()
    ret = httptest(host, ip)
    all_tasks.add((host, ip))
    duration = int((time.time() - t1) * 1000)

    if ret:
        error_task.add((host, ip))
        print '%s(%s) %sms "%s"' % (host, ip, duration, ret)


def run():
    all_tasks.clear()
    for domain_id, domain_name in domain_list():
        for record in record_list(domain_id):
            sub_domain = record['name']
            ip = record['value']
            host = get_host(domain_name, sub_domain)
            test_one(host, ip)

    print  
    print 'total: %s, error: %s' % (len(all_tasks), len(error_task))


if __name__ == '__main__':
    run()
