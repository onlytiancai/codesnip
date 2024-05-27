import random

n = 5000
data = [random.randint(1, n) for i in range(n)]

sum1,sum2 = 0, 0
for i,x in enumerate(data):
    if (i+1) % 1000 == 0:
        print('progress:', i+1)
    if x % 2 == 0:
        sum1 += x
    else:
        sum2 += x
print(f'sum1={sum1},sum2={sum2}')
