import unittest
import textwrap
from tree import *

class MainTest(unittest.TestCase):
    def _read_tree(self):
        txt = textwrap.dedent('''\
        5 3 7
        3 2 4
        7 6 8''')
        tree = read_tree(txt) 
        self.assertTrue(tree is not None)
        return tree

    def test_pre_order_recursion(self):
        tree = self._read_tree()
        output = pre_order_recursion(tree)
        self.assertEqual([5, 3, 2, 4, 7, 6, 8], output)

    def test_pre_order_non_recursion(self):
        tree = self._read_tree()
        output = pre_order_non_recursion(tree)
        self.assertEqual([5, 3, 2, 4, 7, 6, 8], output)

if __name__ == '__main__':
    unittest.main()
