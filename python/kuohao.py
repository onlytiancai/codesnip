txt = '2*3*4*5*6*7'

def match_right(txt):
    print(txt, txt.count('('))

def process(prefix, txt):
    if (prefix+txt).count('(') ==5:
        match_right(prefix+txt)
    if len(txt) < 3:
        return txt 

    for i in range(txt.count('*')):
        process(prefix+ '(' * (i+1) + txt[:2], txt[2:])

process('', txt)
