from functools import reduce
ret = 0
for i in range(100):
    if i%2 == 0:
        ret+=i
print(ret)

print(reduce(lambda x,y: x+y, filter(lambda x: x%2==0, range(100)), 0))
