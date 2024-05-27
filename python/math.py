add = lambda x, y: x + y
sub = lambda x, y: add(x, -y)
def mul(x, y):
    ret = 0
    while True:
        ret = add(ret, x)
        y = sub(y, 1)
        if y == 0:
            break
    return ret
def div(x, y):
    ret = 0
    while True:
        ret = add(ret, 1)
        x = sub(x, y)
        if x <= 0:
            break
    return ret 
def pow(x, y):
    ret = 1
    while True:
        ret = mul(ret, x)
        y = sub(y, 1)
        if y == 0:
            break
    return ret


if __name__ == '__main__':
    print(add(6, 2))
    print(sub(6, 2))
    print(mul(6, 2))
    print(div(6, 2))
    print(pow(2, 4))


