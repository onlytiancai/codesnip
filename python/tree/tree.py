'''
数据结构（一）-- 平衡树
https://www.cnblogs.com/Benjious/p/10336145.html
硬核图解面试最怕的红黑树【建议反复摩擦】
https://blog.csdn.net/qq_35190492/article/details/109503539
'''
import pprint

class Node(object):
    l = None
    r = None
    def __init__(self, v, l=None, r=None):
        self.value = float(v)
        if l is not None and l != '-':
            self.l = Node(l) 
        if r is not None and r != '-':
            self.r = Node(r) 

    def __str__(self, level=0):
        ret = ''
        if level > 0:
            if level > 1:
                ret += '   ' * (level - 1)
            ret += '|--'
        ret += repr(self.value)+"\n"
        if self.l is not None:
            ret += self.l.__str__(level+1)
        if self.r is not None:
            ret += self.r.__str__(level+1)
        return ret

def read_tree(input):
    l_nodes = {}
    r_nodes = {}
    root = None
    for line in input.splitlines():
        v, l, r = line.strip().split() 
        node = Node(v, l, r)
        if not root:
            root = node

        if v in l_nodes:
            l_nodes[v].l = node

        if v in r_nodes:
            r_nodes[v].r = node

        l_nodes[l] = node
        r_nodes[r] = node

    return root

def pre_order_recursion(tree):
    ret = []
    if tree is not None:
        ret.append(tree.value)
        ret.extend(pre_order_recursion(tree.l))
        ret.extend(pre_order_recursion(tree.r))
    return ret

def in_order_recursion(tree):
    ret = []
    if tree is not None:
        ret.extend(in_order_recursion(tree.l))
        ret.append(tree.value)
        ret.extend(in_order_recursion(tree.r))
    return ret

def post_order_recursion(tree):
    ret = []
    if tree is not None:
        ret.extend(in_order_recursion(tree.l))
        ret.extend(in_order_recursion(tree.r))
        ret.append(tree.value)
    return ret

def breadth_first(tree):
    ret = [] 
    queue = [tree]
    while len(queue) > 0:
        node = queue.pop()
        ret.append(node.value)
        if node.l is not None:
            queue.insert(0, node.l)
        if node.r is not None:
            queue.insert(0, node.r)
    return ret;

def depth_first(tree):
    ret = [] 
    stack = [tree]
    while len(stack) > 0:
        node = stack.pop()
        ret.append(node.value)
        if node.r is not None:
            stack.append(node.r)
        if node.l is not None:
            stack.append(node.l)
    return ret;

def pre_order_non_recursion(tree):
    ret = []
    stack = []
    node = tree
    while node is not None or len(stack) > 0:
        if node is not None:
            ret.append(node.value)
            stack.append(node)
            node = node.l
        else:
            node = stack.pop()
            node = node.r

    return ret

def in_order_non_recursion(tree):
    ret = []
    stack = []
    node = tree
    while node is not None or len(stack) > 0:
        if node is not None:
            stack.append(node)
            node = node.l
        else:
            node = stack.pop()
            ret.append(node.value)
            node = node.r

    return ret

def rotate_right(node):
    y = node.l 
    node.l = y.r
    y.r = node
    return y

def rotate_left(node):
    y = node.r 
    node.r = y.l
    y.l = node
    return y

def tree_height(tree):
    if tree is None:
        return 0
    l_height = 0 if tree.l is None else tree_height(tree.l) 
    r_height = 0 if tree.r is None else tree_height(tree.r) 
    return max(l_height, r_height) + 1

def blance_factor(tree):
    if tree is None:
        return 0
    return tree_height(tree.l) - tree_height(tree.r)

def lose_blance_type(tree):
    if blance_factor(tree) > 1 and blance_factor(tree.l) > 0:
        return 'll'
    if blance_factor(tree) > 1 and blance_factor(tree.l) < 0:
        return 'lr'
    if blance_factor(tree) < -1 and blance_factor(tree.r) < 0:
        return 'rr'
    if blance_factor(tree) < -1 and blance_factor(tree.r) > 0:
        return 'rl'
    return '' 

def re_blance(tree):
    if lose_blance_type(tree) == 'll':
        return rotate_right(tree)

    if lose_blance_type(tree) == 'lr':
        tree.l = rotate_left(tree.l)
        return rotate_right(tree)

    if lose_blance_type(tree) == 'rr':
        return rotate_left(tree)

    if lose_blance_type(tree) == 'rl':
        tree.r = rotate_right(tree.r)
        return rotate_left(tree)

    return tree

def put_value(tree, value):
    if tree is None:
        return Node(value)

    if value  < tree.value:
        tree.l = put_value(tree.l, value)
    elif value > tree.value:
        tree.r = put_value(tree.r, value)
    else:
        tree.value = value

    return re_blance(tree)
