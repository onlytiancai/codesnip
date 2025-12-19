#!/usr/bin/env python3
import re
import sys
import time
import argparse
import datetime
import os
from collections import defaultdict, deque

class LogParser:
    def __init__(self, pattern=None):
        # 默认解析模式，匹配提供的日志格式
        if pattern is None:
            pattern = r'^(?P<timestamp>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+\-]\d{2}:\d{2})\s+(?P<status_code>\d+)\s+(?P<response_time>\d+\.\d+)\s+(?P<upstream_time>\d+\.\d+)\s+(?P<unknown1>\S+)\s+(?P<client_ip>\S+)\s+"(?P<x_forwarded_for>[^"]*)"\s+(?P<host>\S+)\s+(?P<upstream_addr>\S+)\s+"(?P<request>[^"]*)"\s+(?P<body_bytes_sent>\d+)\s+"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"$'
        self.pattern = re.compile(pattern)

    def parse(self, line):
        line = line.strip()
        match = self.pattern.match(line)
        if match:
            return match.groupdict()
        return None

class LogProcessor:
    def __init__(self, parser, group_by='minute', aggregations=None):
        self.parser = parser
        self.group_by = group_by
        self.aggregations = aggregations or []
        self.data = defaultdict(lambda: defaultdict(int))
        self.sums = defaultdict(lambda: defaultdict(float))
        self.counts = defaultdict(lambda: defaultdict(int))

    def _get_time_key(self, timestamp_str):
        dt = datetime.datetime.fromisoformat(timestamp_str)
        if self.group_by == 'minute':
            return dt.strftime('%Y-%m-%d %H:%M')
        elif self.group_by == 'hour':
            return dt.strftime('%Y-%m-%d %H')
        elif self.group_by == 'day':
            return dt.strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def process_line(self, line):
        parsed = self.parser.parse(line)
        if not parsed:
            return

        time_key = self._get_time_key(parsed['timestamp'])

        for agg in self.aggregations:
            if agg['type'] == 'count':
                if 'filter' in agg:
                    if parsed.get(agg['filter']['field']) == agg['filter']['value']:
                        self.data[time_key][agg['name']] += 1
                else:
                    self.data[time_key][agg['name']] += 1
            elif agg['type'] == 'sum':
                if agg['field'] in parsed:
                    try:
                        value = float(parsed[agg['field']])
                        self.data[time_key][agg['name']] += value
                    except ValueError:
                        pass
            elif agg['type'] == 'avg':
                if agg['field'] in parsed:
                    try:
                        value = float(parsed[agg['field']])
                        self.sums[time_key][agg['name']] += value
                        self.counts[time_key][agg['name']] += 1
                        self.data[time_key][agg['name']] = self.sums[time_key][agg['name']] / self.counts[time_key][agg['name']]
                    except ValueError:
                        pass

    def get_results(self):
        return dict(self.data)

class LogReader:
    def __init__(self, file_path, follow=False, processor=None):
        self.file_path = file_path
        self.follow = follow
        self.processor = processor

    def read(self):
        with open(self.file_path, 'r') as f:
            # 先读取现有内容
            for line in f:
                if self.processor:
                    self.processor.process_line(line)
                yield line

            # 如果是follow模式，继续监听新内容
            if self.follow:
                while True:
                    where = f.tell()
                    line = f.readline()
                    if not line:
                        time.sleep(0.1)
                        f.seek(where)
                    else:
                        if self.processor:
                            self.processor.process_line(line)
                        yield line

def parse_aggregation(agg_str):
    # 解析聚合表达式，格式如：count(status_code=200),sum(body_bytes_sent),avg(response_time)
    aggregations = []
    parts = agg_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if '(' in part and ')' in part:
            agg_type, rest = part.split('(', 1)
            rest = rest.rstrip(')')
            agg_name = f"{agg_type}_{rest}"
            
            if '=' in rest:
                field, value = rest.split('=', 1)
                aggregations.append({
                    'type': agg_type,
                    'name': agg_name,
                    'filter': {'field': field, 'value': value}
                })
            else:
                aggregations.append({
                    'type': agg_type,
                    'name': agg_name,
                    'field': rest
                })
    return aggregations

def main():
    parser = argparse.ArgumentParser(description='Wawa Log Viewer - 一个灵活的日志分析工具')
    parser.add_argument('logfile', help='日志文件路径')
    parser.add_argument('--pattern', help='自定义日志解析正则表达式')
    parser.add_argument('--group-by', choices=['second', 'minute', 'hour', 'day'], default='minute', help='时间分组粒度')
    parser.add_argument('--aggregate', help='聚合表达式，如：count(status_code=200),sum(body_bytes_sent),avg(response_time)')
    parser.add_argument('--tail', action='store_true', help='启用tail -f模式实时监控日志')
    parser.add_argument('--interval', type=int, default=5, help='tail模式下的结果输出间隔（秒）')
    
    args = parser.parse_args()

    # 创建日志解析器
    log_parser = LogParser(args.pattern)

    # 创建聚合器
    aggregations = []
    if args.aggregate:
        aggregations = parse_aggregation(args.aggregate)
    
    processor = LogProcessor(log_parser, group_by=args.group_by, aggregations=aggregations)

    # 创建日志读取器
    reader = LogReader(args.logfile, follow=args.tail, processor=processor)

    if args.tail:
        # tail模式，定期输出结果
        print(f"实时监控日志文件: {args.logfile}")
        print(f"分组粒度: {args.group_by}")
        if aggregations:
            print(f"聚合规则: {args.aggregate}")
        print("=" * 60)
        
        last_print = time.time()
        for line in reader.read():
            current_time = time.time()
            if current_time - last_print >= args.interval:
                results = processor.get_results()
                if results:
                    print(f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 统计结果:")
                    for time_key in sorted(results.keys()):
                        print(f"{time_key}: {results[time_key]}")
                last_print = current_time
    else:
        # 一次性处理模式
        print(f"分析日志文件: {args.logfile}")
        print(f"分组粒度: {args.group_by}")
        if aggregations:
            print(f"聚合规则: {args.aggregate}")
        print("=" * 60)
        
        for line in reader.read():
            pass
        
        results = processor.get_results()
        if results:
            for time_key in sorted(results.keys()):
                print(f"{time_key}: {results[time_key]}")
        else:
            print("没有解析到符合格式的日志行")

if __name__ == '__main__':
    main()
