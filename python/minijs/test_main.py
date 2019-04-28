# -*- coding: utf-8 -*-
import unittest
import minijs


class ParserTest(unittest.TestCase):
    def test0(self):
        self.assertListEqual(['1'], minijs.parse_tokens('1'))

    def test1(self):
        self.assertListEqual(['1', '+', '2'], minijs.parse_tokens('1 + 2'))

    def test2(self):
        expect = ['(', '1', '+', '2', ')', '*', '(', '3', '+', '4', ')']
        actual = minijs.parse_tokens('(1+2)*(3+4)')
        self.assertListEqual(expect, actual)


class MyTest(unittest.TestCase):
    def test0(self):
        self.assertEqual('1', minijs.eval('1'), '纯数字')

    def test1(self):
        self.assertEqual('2', minijs.eval('1 + 1'), '加法')


if __name__ == '__main__':
    unittest.main()
