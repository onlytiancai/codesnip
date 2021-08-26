import unittest
from tree import *

class MainTest(unittest.TestCase):
    def test_main(self):
        txt = '2 1 3'
        tree = read_tree(txt) 
        self.assertTrue(tree is not None)
        output = print_tree(tree)
        self.assertEqual([1, 2, 3], output)

if __name__ == '__main__':
    unittest.main()
