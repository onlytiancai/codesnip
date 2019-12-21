# -*- coding: utf-8 -*-
from collections import defaultdict
import time

words = []
t1 = time.time()
for line in open('c:/allfiles.txt', encoding="utf-8"):
    words.append(line)
print('read words take time:', (time.time() - t1)*1000) 
print("words:",len(words))
   
t1 = time.time()    
char2map = defaultdict(set)
char1map = defaultdict(set)
for word in words:
    for i in range(len(word)):
        char1map[word[i]].add(word)
        if i > 0:
            char2map[word[i-1:i+1]].add(word)
                
print('build index take time:', (time.time() - t1)*1000)
print("char1map:",len(char1map))
print("char2map:",len(char2map))


s = 'test'
t1 = time.time()
set_list = (char1map.get(c, set()) for c in s)
u = set.intersection(*set_list)
print('intersection', len(u))

ret = [item for item in u if item.find(s) != -1]
print('char1 search take time:', (time.time() - t1)*1000, len(ret))

t1 = time.time()
set_list = (char2map.get(s[i-2:i], set()) for i in range(2,len(s)+1))
u = set.intersection(*set_list)
print('intersection', len(u))
print('char2 search take time:', (time.time() - t1)*1000, len(ret))

t1 = time.time()
ret = [item for item in words if item.find(s) != -1]
print('full search take time:', (time.time() - t1)*1000, len(set(ret)))

