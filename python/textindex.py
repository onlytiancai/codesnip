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
    chars = list(set(word))
    for i in range(len(chars)):
        char1map[chars[i]].add(word)

for word in words:
    for i in range(1, len(word)):
        k = ''.join(word[i-1:i+1])
        char2map[k].add(word)
                
print('build index take time:', (time.time() - t1)*1000)
print("char1map:",len(char1map))
print("char2map:",len(char2map))


s = 'test'
t1 = time.time()
set_list = (char1map.get(c, set()) for c in s)
u = set.intersection(*set_list)
print('char1 intersection', len(u))

ret = [item for item in u if item.find(s) != -1]
print('char1 search take time:', (time.time() - t1)*1000, len(ret))

t1 = time.time()
for i in range(len(s)):
    if i > 0:
        k = s[i-1:i+1]
        l = char2map.get(k, set())
        print(k, len(l))
set_list = (char2map.get(s[i-1:i+1], set()) for i in range(len(s)) if i > 0)
u = set.intersection(*set_list)
print('char2 intersection', len(u))

ret = [item for item in u if item.find(s) != -1]
print('char2 search take time:', (time.time() - t1)*1000, len(ret))

t1 = time.time()
ret = [item for item in words if item.find(s) != -1]
print('full search take time:', (time.time() - t1)*1000, len(set(ret)))

