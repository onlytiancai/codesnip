'''
左耳朵耗子的csv多列排序题目
http://weibo.com/1401880315/zd7nBxyDi
'''
import csv
import sys

map(csv.writer(sys.stdout).writerow, sorted(csv.reader(open('./input.csv', 'rb'))))
