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
    print(input)
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

def depth_first(tree):
    ret = []
    ret.append(tree.value)
    if tree.l is not None:
        ret.extend(depth_first(tree.l))
    if tree.r is not None:
        ret.extend(depth_first(tree.r))
    return ret
