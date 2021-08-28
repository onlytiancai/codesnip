import unittest
import textwrap
from tree import *

class MainTest(unittest.TestCase):
    def test_depth_first(self):
        txt = textwrap.dedent('''\
        5 3 7
        3 2 4
        7 6 8''')
        tree = read_tree(txt) 
        self.assertTrue(tree is not None)
        output = depth_first(tree)
        self.assertEqual([5, 3, 2, 4, 7, 6, 8], output)

if __name__ == '__main__':
    unittest.main()
