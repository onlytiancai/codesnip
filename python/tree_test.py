import unittest
import textwrap
from tree import *

class MainTest(unittest.TestCase):
    '''
       5
     3   7
    2 4 6 8
    '''
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

    def test_in_order_recursion(self):
        tree = self._read_tree()
        output = in_order_recursion(tree)
        self.assertEqual([2, 3, 4, 5, 6, 7, 8], output)

    def test_post_order_recursion(self):
        tree = self._read_tree()
        output = post_order_recursion(tree)
        self.assertEqual([2, 3, 4, 6, 7, 8, 5], output)

    def test_breadth_first(self):
        tree = self._read_tree()
        output = breadth_first(tree)
        self.assertEqual([5, 3, 7, 2, 4, 6, 8], output)

    def test_depth_first(self):
        tree = self._read_tree()
        output = depth_first(tree)
        self.assertEqual([5, 3, 2, 4, 7, 6, 8], output)

    def test_pre_order_non_recursion(self):
        tree = self._read_tree()
        output = pre_order_non_recursion(tree)
        self.assertEqual([5, 3, 2, 4, 7, 6, 8], output)

    def test_in_order_non_recursion(self):
        tree = self._read_tree()
        output = in_order_non_recursion(tree)
        self.assertEqual([2, 3, 4, 5, 6, 7, 8], output)

    def test_rotate_right(self):
        tree = read_tree(textwrap.dedent('''\
        3 2 - 
        2 1 -''')) 
        self.assertTrue(tree is not None)
        
        output = breadth_first(tree)
        self.assertEqual([3, 2, 1], output)

        tree = rotate_right(tree)
        output = breadth_first(tree)
        self.assertEqual([2, 1, 3], output)

    def test_rotate_right_with_move_left_child(self):
        tree = read_tree(textwrap.dedent('''\
        5 3 6 
        3 2 4
        2 1 -''')) 
        self.assertTrue(tree is not None)
        
        output = breadth_first(tree)
        self.assertEqual([5, 3, 6, 2, 4, 1], output)

        tree = rotate_right(tree)
        output = breadth_first(tree)
        self.assertEqual([3, 2, 5, 1, 4, 6], output)

    def test_rotate_left(self):
        tree = read_tree(textwrap.dedent('''\
        1 - 2 
        2 - 3''')) 
        self.assertTrue(tree is not None)
        
        output = breadth_first(tree)
        self.assertEqual([1, 2, 3], output)

        tree = rotate_left(tree)
        output = breadth_first(tree)
        self.assertEqual([2, 1, 3], output)

    def test_str_tree(self):
        tree = self._read_tree()
        s = str(tree)
        self.assertEqual(textwrap.dedent('''\
        5.0
        |--3.0
           |--2.0
           |--4.0
        |--7.0
           |--6.0
           |--8.0
        '''), s)

    def test_tree_height(self):
        tree = self._read_tree()
        self.assertEqual(3, tree_height(tree))
        self.assertEqual(2, tree_height(tree.l))
        self.assertEqual(1, tree_height(tree.l.l))
        self.assertEqual(0, tree_height(tree.l.l.l))

    def test_balance_factor(self):
        tree = read_tree(textwrap.dedent('''\
        5 3 6 
        3 2 4
        2 1 -''')) 
        self.assertEqual(2, blance_factor(tree))
        self.assertEqual(0, blance_factor(tree.r))
        self.assertEqual(1, blance_factor(tree.l))
        self.assertEqual(1, blance_factor(tree.l.l))
        self.assertEqual(0, blance_factor(tree.l.r))
        self.assertEqual(0, blance_factor(tree.l.l.l))

    def test_lose_bance_type(self):
        tree = read_tree(textwrap.dedent('''\
        1 2 -
        2 3 -
        ''')) 
        self.assertEqual('ll', lose_blance_type(tree))

        tree = read_tree(textwrap.dedent('''\
        3 1 -
        1 - 2 
        ''')) 
        self.assertEqual('lr', lose_blance_type(tree))

        tree = read_tree(textwrap.dedent('''\
        1 - 2
        2 - 3 
        ''')) 
        self.assertEqual('rr', lose_blance_type(tree))

        tree = read_tree(textwrap.dedent('''\
        1 - 2
        2 3 - 
        ''')) 
        self.assertEqual('rl', lose_blance_type(tree))

    def test_re_blance_ll(self):
        tree = read_tree(textwrap.dedent('''\
        3 2 -
        2 1 -
        ''')) 
        tree = re_blance(tree)
        self.assertEqual([2, 1, 3], breadth_first(tree))

    def test_re_blance_lr(self):
        tree = read_tree(textwrap.dedent('''\
        3 1 -
        1 - 2
        ''')) 
        tree = re_blance(tree)
        self.assertEqual([2, 1, 3], breadth_first(tree))

    def test_re_blance_rr(self):
        tree = read_tree(textwrap.dedent('''\
        1 - 2 
        2 - 3 
        ''')) 
        tree = re_blance(tree)
        self.assertEqual([2, 1, 3], breadth_first(tree))

    def test_re_blance_rl(self):
        tree = read_tree(textwrap.dedent('''\
        1 - 3 
        3 2 - 
        ''')) 
        tree = re_blance(tree)
        self.assertEqual([2, 1, 3], breadth_first(tree))

    def test_put_value(self):
        tree = read_tree(textwrap.dedent('''\
        2 1 3
        ''')) 
        tree = put_value(tree, 0)
        self.assertEqual([2, 1, 3, 0], breadth_first(tree))

        tree = put_value(tree, 4)
        self.assertEqual([2, 1, 3, 0, 4], breadth_first(tree))

        # rr
        tree = put_value(tree, 5)
        self.assertEqual([2, 1, 4, 0, 3, 5], breadth_first(tree))

        # rl 
        tree = put_value(tree, 6)
        tree = put_value(tree, 5.5)
        self.assertEqual([2, 1, 4, 0, 3, 5.5, 5, 6], breadth_first(tree))

        # ll
        tree = put_value(tree, -1)
        self.assertEqual([2, 0, 4, -1, 1, 3, 5.5, 5, 6], breadth_first(tree))
        
        # lr
        tree = put_value(tree, -2)
        tree = put_value(tree, -1.5)
        self.assertEqual([2, 0, 4, -1.5, 1, 3, 5.5, -2, -1, 5, 6], breadth_first(tree))
if __name__ == '__main__':
    unittest.main()
