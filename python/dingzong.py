def f(x,i=1):
    if x < 0:
        return -x
    return 0.5*f(x-f(x-1))

print('f(0)=',f(0))
print('f(1)=',f(1))
print('f(2)=',f(2))
print('f(3)=',f(3))
