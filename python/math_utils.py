# -*- coding:utf-8 -*-
'''
Python调试及单元测试 http://www.fuzhijie.me/?p=310
'''

def add(op1, op2):
    return op1+op2
 
def sub(op1, op2):
    return op1-op2

import unittest

class MathTestCase(unittest.TestCase):
    '''math test case'''
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def testAdd(self):
        '''test add'''
        self.assertEqual(add(4, 5), 9)
    def testSub(self):
        '''test sub'''
        self.assertEqual(sub(8, 5), 3)
    def suite():
        suite = unittest.TestSuite()
        suite.addTest(MathTestCase())
        return suite
if __name__ == '__main__':
    unittest.main()
