import random
from uuid import uuid4

with open('test_data.csv', 'a') as f:
    for i in range(1000000):
        s = '%s,%s\n' % (random.randint(1,2000000), uuid4())
        f.write(s)
        print(i, s,end='')
