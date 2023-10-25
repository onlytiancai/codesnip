import re
def parse(sql):
    ret = {}
    tokens = re.findall(r'([\w\(\)*]+|=|"[^"]+"|\'[^\']+\')',sql)
    i, l = 0, len(tokens)
    while i < l:
        if tokens[i] == 'select':
            selected = []
            i += 1
            while tokens[i] != 'from':
                selected.append(tokens[i])
                i += 1
            ret['selected'] = selected
        if tokens[i] == 'from':
            i += 1
            ret['table'] = tokens[i]
            i += 1
        if i<l and tokens[i] == 'where':
            where = {}
            i += 1
            while i < l and tokens[i] not in ('group', 'order'):
                key = tokens[i] 
                i += 1
                if tokens[i] not in ('='):
                    raise Exception('unexpect token:', tokens[i])
                i += 1
                value = tokens[i]
                i += 1
                if i<l and tokens[i] == 'and':
                    i += 1 # just skip
                where[key] = value
            ret['where'] = where
        if i<l and tokens[i] == 'group':
            i += 1
            if tokens[i] != 'by':
                raise Exception('unexpect token:', tokens[i])
            i += 1
            ret['group'] = tokens[i] 
            i += 1
        if i<l and tokens[i] == 'order':
            i += 1
            if tokens[i] != 'by':
                raise Exception('unexpect token:', tokens[i])
            i += 1
            ret['order'] = tokens[i] 
            i += 1
            if i<l and tokens[i] == 'desc':
                ret['order'] = ret['order'] + ' desc'

        i += 1
    return ret 

if __name__ == '__main__':
    'select id,name from t1'
    'select id,name from t1 where id=3'
    'select id,name from t1 where id=3 order by id desc'
    'select id,name,age,count(*) from t1 where id=3 and name="4 5" and  group by id order by name desc'
    import sys
    ast = parse(sys.argv[1])
    from pprint import pprint
    pprint(ast)

