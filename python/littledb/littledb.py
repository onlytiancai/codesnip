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
        if os.path.exists(self.db_path):
            raise Exception('%s is exist' % self.db_path)
        os.mkdir(self.db_path)

    def create_table(self, table_name):
        table = Table(self, table_name)
        table.create()
        return table


class Table(object):
    '数据表'
    def __init__(self, db, table_name):
        self.db = db
        self.table_name = table_name
        self.table_path = os.path.join(self.db.db_path, table_name + '.data')

    def create(self):
        '创建表'
        logging.debug('table_path:%s', self.table_path)
        if os.path.exists(self.table_path):
            raise Exception('%s is exist' % self.table_path)
        open(self.table_path, 'a').close()

    def insert(self, key, row):
        '插入数据'
        if not isinstance(key, int):
            raise Exception('key must be int')

        line = '%s\t%s\n' % (key, json.dumps(row))
        with open(self.table_path, 'a') as f:
            f.write(line)

    def find(self, key):
        '按主键查找数据'
        if not isinstance(key, int):
            raise Exception('key must be int')

        with open(self.table_path, 'r') as f:
            for line in f:
                row_key, row = line.split('\t')
                if int(row_key) == key:
                    return json.loads(row)

def create_db(db_path):
    '创建数据库'
    db = Db(db_path)
    db.create()
    return db
