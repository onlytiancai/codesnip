#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
各监控指标地址：http://wenku.baidu.com/view/ac9322896f1aff00bed51ed6.html
'''

import MySQLdb
import time
import socket

api_key = '5cdd8568-9268-4b2d-8cf4-184254132aaf'
host = 'mysql'
ip = '127.0.0.1'
conn_args = dict(host="localhost", user="root", passwd="password", db="test", charset="utf8")
interval = 5
collector_url = ('collector.monitor.dnspod.cn', 2003)

global_status = {}
global_status2 = {}

client = socket.socket()
client.connect(collector_url)


def get_mysql_status():
    conn = MySQLdb.connect(**conn_args)
    cursor = conn.cursor()

    sql = 'show global status;'
    cursor.execute(sql, ())
    ret = dict((row[0].lower(), row[1]) for row in cursor.fetchall())

    cursor.close()
    conn.close()
    return ret


def get_metric(name):
    name = name.lower().strip()
    return int(global_status.get(name, -1))


def get_metric2(name):
    name = name.lower().strip()
    return int(global_status2.get(name, -1))

def send_metric(name, value):
    data = '%(api_key)s/%(host)s/%(ip)s/%(metric_name)s %(value)s %(timestamp)s\n'
    data = data % dict(api_key=api_key, host=host, ip=ip,
                       metric_name=name, value=value, timestamp=time.time())
    client.send(data)

def show_metric(name, value):
    print name, value
    send_metric(name, value)


def connection_status():
    show_metric('Max_used_connections', get_metric('Max_used_connections'))


def qps_status():
    qps_select = (get_metric2('Com_select') - get_metric('Com_select')) / interval
    qps_insert = (get_metric2('Com_insert') - get_metric('Com_insert')) / interval
    qps_update = (get_metric2('Com_update') - get_metric('Com_update')) / interval
    qps_delete = (get_metric2('Com_delete') - get_metric('Com_delete')) / interval
    qps_iud = qps_insert + qps_update + qps_delete
    show_metric('qps_select', qps_select)
    show_metric('qps_iud', qps_iud)


def slow_query_status():
    show_metric('Slow_launch_threads', get_metric('Slow_launch_threads'))
    show_metric('Slow_queries', get_metric('Slow_queries'))


def key_buffer_status():
    key_reads = get_metric('Key_reads')
    key_read_requests = get_metric('Key_read_requests')
    key_writes = get_metric('Key_writes')
    key_write_requests = get_metric('Key_write_requests')
    
    if key_read_requests != 0:
        key_buffer_read_hits = round((1 - key_reads / float(key_read_requests)) * 100, 2)
    else:
        key_buffer_read_hits = 0

    if key_write_requests != 0:
        key_buffer_write_hits = round((1 - key_writes / float(key_write_requests)) * 100, 2)
    else:
        key_buffer_write_hits = 0

    show_metric('key_buffer_read_hits', key_buffer_read_hits)
    show_metric('key_buffer_write_hits', key_buffer_write_hits)


def innodb_buffer_status():
    innodb_buffer_pool_reads = get_metric('innodb_buffer_pool_reads')
    innodb_buffer_pool_read_requests = get_metric('innodb_buffer_pool_read_requests')

    if innodb_buffer_pool_read_requests != 0:
        innodb_buffer_read_hits = round((1 - innodb_buffer_pool_reads / float(innodb_buffer_pool_read_requests)) * 100, 2)
    else:
        innodb_buffer_read_hits = 0

    show_metric('innodb_buffer_read_hits', innodb_buffer_read_hits)


def qcache_status():
    qcahce_hits = get_metric('Qcache_hits')
    qcache_inserts = get_metric('Qcache_inserts')

    if qcahce_hits + qcache_inserts != 0:
        query_cache_hits = round((float(qcahce_hits) / (qcahce_hits + qcache_inserts)) * 100, 2)
    else:
        query_cache_hits = 0

    show_metric('query_cache_hits', query_cache_hits)


def thread_cache_status():
    threads_created = get_metric('Threads_created')
    connections = get_metric('Connections')
    if connections != 0:
        thread_cache_hits = round((1 - threads_created / float(connections)) * 100, 2)
    else:
        thread_cache_hits = 0
    show_metric('thread_cache_hits', thread_cache_hits)
 

if __name__ == '__main__':
    global_status = get_mysql_status()
    print 'please wait %s seconds' % interval
    time.sleep(interval)
    global_status2 = get_mysql_status()

    connection_status()
    qps_status()
    slow_query_status()
    key_buffer_status()
    innodb_buffer_status()
    qcache_status()
    thread_cache_status()

    client.close()
