# -*- coding:utf-8 -*-
'展示垃圾都是什么东西'
#《Python Cookbook》p319

import gc


def dump_garbage():
    '展示垃圾都是什么东西'
    print "\nGARBAGE:"
    gc.collect()
    print "\nGARBAGE OBJECTS:"
    for x in gc.garbage:
        s = str(x)
        if len(s) > 80:
            s = s[:77] + '...'
        print type(x), '\n ', s

if __name__ == '__main__':
    gc.enable()
    gc.set_debug(gc.DEBUG_LEAK)
    
    #模拟一个泄露（一个引用自身的列表）并展示
    l = []
    l.append(l)
    del l
    dump_garbage()
