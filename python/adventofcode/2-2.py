total = 0
for line in open('data2.txt'):
    line = line.strip()
    title, items = line.split(':')
    id = int(title.split(' ')[1])
    items = items.split(';')
    print(id, items)
    max_map = {'red': 0, 'green': 0, 'blue':0}
    for item in items:
        arr = item.split(', ')
        ok = True
        for x in arr:
            n, color = x.strip().split(' ')
            if int(n) > max_map[color]:
                max_map[color] = int(n)
    current = 1
    for x in max_map:
        current *= max_map[x]
    print(max_map, current)
    total+=current


print(total)
