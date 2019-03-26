#! /usr/bin/env python3
#-*- coding: utf-8 -*-

import os
import sys
import json
import logging

logging.basicConfig(format='%(asctime)-15s %(message)s', level=logging.NOTSET, stream=sys.stdout)

class Db(object):
    '数据库'
    def __init__(self, db_path):
        self.db_path = db_path

    def create(self):
        '创建库'
        if os.path.exists(self.db_path):
            raise Exception('%s is exist' % self.db_path)
        os.mkdir(self.db_path)

    def create_table(self, table_name):
        '创建表'
        table = Table(self, table_name)
        table.create()
        return table

    def load(self):
        '加载库'
        if not os.path.exists(self.db_path):
            raise Exception('%s is not exist' % self.db_path)

    def load_table(self, table_name):
        '加载表'
        table = Table(self, table_name)
        table.load()
        return table

class Table(object):
    '数据表'
    def __init__(self, db, table_name):
        self.db = db
        self.table_name = table_name
        self.table_path = os.path.join(self.db.db_path, table_name + '.data')
        self.index_path = os.path.join(self.db.db_path, table_name + '.index')
        self.index = {}

    def create(self):
        '创建表'
        logging.debug('create table:%s', self.table_path)
        if os.path.exists(self.table_path):
            raise Exception('%s is exist' % self.table_path)
        open(self.table_path, 'a').close()

    def load(self):
        '加载表'
        logging.debug('load table:%s', self.table_path)
        if not os.path.exists(self.table_path):
            raise Exception('%s is not exist' % self.table_path)
        self._load_index()

    def upinsert(self, key, row):
        '插入数据'
        if not isinstance(key, int):
            raise Exception('key must be int')

        line = '%s\t%s\n' % (key, json.dumps(row))
        with open(self.table_path, 'a') as f:
            self._index(key, f.tell())
            f.write(line)

    def find(self, key):
        '按主键查找数据'
        if not isinstance(key, int):
            raise Exception('key must be int')

        offset = self._find_offset(key)

        with open(self.table_path, 'r') as f:
            f.seek(offset)
            line = f.readline()
            row_key, row = line.split('\t')
            if int(row_key) == key:
                return json.loads(row)

    def _find_offset(self, key):
        '索引中查找数据位置'
        if key not in self.index:
            raise Exception('%s not found') 
        return self.index[key]

    def _index(self, key, offset):
        '更新索引'
        self.index[key] = offset
        line = '%s\t%s\n' % (key, offset)
        with open(self.index_path, 'a') as f:
            f.write(line)

    def _load_index(self):
        '加载索引'
        with open(self.index_path, 'r') as f:
            for line in f:
                row_key, offset = line.split('\t')
                self.index[int(row_key)] = int(offset)

def create_db(db_path):
    '创建数据库'
    db = Db(db_path)
    db.create()
    return db

def load_db(db_path):
    '加载数据库'
    db = Db(db_path)
    db.load()
    return db
