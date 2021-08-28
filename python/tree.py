import pprint

class Node(object):
    l = None
    r = None
    def __init__(self, v, l=None, r=None):
        self.value = int(v)
        if l is not None:
            self.l = Node(l) 
        if r is not None:
            self.r = Node(r) 

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



