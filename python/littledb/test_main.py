#! /usr/bin/env python3
#-*- coding: utf-8 -*-
import unittest

import littledb

class TestDb(unittest.TestCase):
    db_path = './mydb'
    table = 'test'

    def setUp(self):
        print('setUp ...')
        import os
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def test_create_table(self):
        '测试新建表'
        db, table = self._create_table()
        self._test_find(table)
        self._test_update(table)

    def test_load_table(self):
        '测试加载表'
        self._create_table()

        db = littledb.load_db(self.db_path)
        table = db.load_table(self.table)
        self._test_find(table)
        self._test_update(table)

    def _create_table(self):
        '建库，建表，插入测试数据'
        db = littledb.create_db(self.db_path)
        table = db.create_table(self.table)

        table.upinsert(1, {'a': 1, 'b': 2})
        table.upinsert(2, {'a': 3, 'b': 4})
        return db, table

    def _test_find(self, table):
        '测试基本查找'
        row = table.find(1)
        self.assertEqual(row['a'], 1)
        self.assertEqual(row['b'], 2)

        row = table.find(2)
        self.assertEqual(row['a'], 3)
        self.assertEqual(row['b'], 4)

    def _test_update(self, table):
        '测试更新后查找'
        table.upinsert(1, {'a': 5, 'b': 6})
        row = table.find(1)
        self.assertEqual(row['a'], 5)
        self.assertEqual(row['b'], 6)


if __name__ == '__main__':
    unittest.main()
