#! /usr/bin/env python3
#-*- coding: utf-8 -*-
import unittest

import littledb

class TestDb(unittest.TestCase):
    db_path = './mydb'

    def setUp(self):
        import os
        import shutil
        if os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)

    def tearDown(self):
        print('tearDown...')

    def test_main(self):
        db = littledb.create_db(self.db_path)
        table = db.create_table('test')

        table.insert(1, {'a': 1, 'b': 2})
        table.insert(2, {'a': 3, 'b': 4})
        table.insert(3, {'a': 5, 'b': 6})

        row = table.find(1)

if __name__ == '__main__':
    unittest.main()
