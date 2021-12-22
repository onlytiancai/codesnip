txt = '2*3*4*5*6*7'

temp = ''
for i, c in enumerate(txt):
    if c.isdigit():
        if i % 4 == 0:
            temp += '(' + c
        else:
            temp += c + ')'
    else:
        temp += c
print(temp)

temp = ''
for i, c in enumerate(txt):
    temp += c
    if i> 0 and c.isdigit():
        temp = '(' + temp + ')'
print(temp)


