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
        self.current_time_window = None
        # 只保存当前时间窗口的数据
        self.data = defaultdict(int)
        self.sums = defaultdict(float)
        self.counts = defaultdict(int)
        self.has_processed_logs = False
        self.headers_printed = False
        # 收集所有聚合字段名称
        self.aggregation_names = [agg['name'] for agg in self.aggregations]
        # 计算每个字段的最大宽度
        self.field_widths = {}
        for name in self.aggregation_names:
            # 初始宽度为字段名长度
            self.field_widths[name] = max(len(name), 10)  # 至少10个字符
        # 时间窗口字段宽度
        self.time_width = 16

    def _get_time_key(self, timestamp_str):
        dt = datetime.datetime.fromisoformat(timestamp_str)
        if self.group_by == 'minute':
            return dt.strftime('%Y-%m-%d %H:%M')
        elif self.group_by == 'hour':
            return dt.strftime('%Y-%m-%d %H')
        elif self.group_by == 'day':
            return dt.strftime('%Y-%m-%d')
        return dt.strftime('%Y-%m-%d %H:%M:%S')

    def _switch_time_window(self, new_time_key):
        """切换时间窗口，输出旧窗口结果并清理数据"""
        if self.current_time_window is not None:
            self._output_current_window()
            self._cleanup_current_window()
        self.current_time_window = new_time_key

    def _output_current_window(self):
        """输出当前时间窗口的结果（横向表格行）"""
        if not self.data and not self.sums:
            return
            
        # 打印表头（仅首次输出时）
        self._print_headers()
        
        # 构建表格行
        row_line = f"{self.current_time_window:<{self.time_width}}"  # 时间窗口列
        
        for name in self.aggregation_names:
            if name in self.counts and self.counts[name] > 0:
                # 平均值（使用sums和counts计算）
                value = f"{self.sums[name] / self.counts[name]:.3f}"
            elif name in self.data:
                # 计数或求和
                value = str(self.data[name])
            else:
                # 该时间窗口没有该字段的数据
                value = "0"
            
            # 确保值的长度不超过字段宽度
            if len(value) > self.field_widths[name]:
                value = value[:self.field_widths[name]]
                
            row_line += f"| {value:<{self.field_widths[name]}}"  # 聚合字段列
        
        # 输出表格行
        print(row_line)

    def _cleanup_current_window(self):
        """清理当前时间窗口的数据"""
        self.data.clear()
        self.sums.clear()
        self.counts.clear()
    
    def _print_headers(self):
        """输出表格表头"""
        if not self.aggregation_names or self.headers_printed:
            return
            
        # 打印表头
        header_line = f"{'TIME WINDOW':<{self.time_width}}"  # 时间窗口列
        for name in self.aggregation_names:
            header_line += f"| {name:<{self.field_widths[name]}}"  # 聚合字段列
        print(header_line)
        
        # 打印分隔线
        separator_line = f"{'=' * self.time_width}"  # 时间窗口分隔线
        for name in self.aggregation_names:
            separator_line += f"|{'=' * (self.field_widths[name] + 1)}"  # 聚合字段分隔线（包含分隔符后的空格）
        print(separator_line)
        
        self.headers_printed = True

    def process_line(self, line):
        parsed = self.parser.parse(line)
        if not parsed:
            return

        self.has_processed_logs = True
        time_key = self._get_time_key(parsed['timestamp'])

        # 检查是否需要切换时间窗口
        if self.current_time_window != time_key:
            self._switch_time_window(time_key)

        for agg in self.aggregations:
            if agg['type'] == 'count':
                if 'filter' in agg:
                    if parsed.get(agg['filter']['field']) == agg['filter']['value']:
                        self.data[agg['name']] += 1
                else:
                    self.data[agg['name']] += 1
            elif agg['type'] == 'sum':
                if agg['field'] in parsed:
                    try:
                        value = float(parsed[agg['field']])
                        self.data[agg['name']] += int(value)
                    except ValueError:
                        pass
            elif agg['type'] == 'avg':
                if agg['field'] in parsed:
                    try:
                        value = float(parsed[agg['field']])
                        self.sums[agg['name']] += value
                        self.counts[agg['name']] += 1
                        # 直接将平均值存储在self.data中
                        self.data[agg['name']] = self.sums[agg['name']] / self.counts[agg['name']]
                    except ValueError:
                        pass

    def get_results(self):
        """获取当前时间窗口的结果"""
        return {self.current_time_window: dict(self.data)} if self.current_time_window else {}

    def flush(self):
        """手动刷新当前时间窗口的结果"""
        if self.current_time_window:
            self._output_current_window()
            self._cleanup_current_window()
            self.current_time_window = None

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
        # tail模式，自动输出结果
        print(f"实时监控日志文件: {args.logfile}")
        print(f"分组粒度: {args.group_by}")
        if aggregations:
            print(f"聚合规则: {args.aggregate}")
        print("=" * 60)
        print("日志分析结果将按时间窗口自动输出:")
        
        for line in reader.read():
            # 所有处理和输出都由processor自动完成
            pass
    else:
        # 一次性处理模式
        print(f"分析日志文件: {args.logfile}")
        print(f"分组粒度: {args.group_by}")
        if aggregations:
            print(f"聚合规则: {args.aggregate}")
        print("=" * 60)
        print("日志分析结果:")
        
        for line in reader.read():
            # 所有处理都由processor自动完成
            pass
        
        # 输出最后一个时间窗口的结果
        processor.flush()
        
        # 检查是否有任何日志被解析
        if not processor.has_processed_logs:
            print("没有解析到符合格式的日志行")

if __name__ == '__main__':
    main()
