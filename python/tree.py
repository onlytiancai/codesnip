class Node(object):
    def __init__(self, n, l, r):
        self.n = int(n)
        self.l = int(l)
        self.r = int(r)

def read_tree(input):
    for line in input.splitlines():
        n, l, r = line.strip().split() 
        return Node(n, l, r)

def print_tree(tree):
    node = tree
    ret = [] 
    ret.extend([node.l, node.n, node.r])
    return ret
