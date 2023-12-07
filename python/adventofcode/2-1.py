total = 0
max_map = {'red': 12, 'green': 13, 'blue':14}
for line in open('data2.txt'):
    line = line.strip()
    title, items = line.split(':')
    id = int(title.split(' ')[1])
    items = items.split(';')
    print(id, items)
    ok_count = 0
    for item in items:
        arr = item.split(', ')
        ok = True
        for x in arr:
            n, color = x.strip().split(' ')
            print('\t', n, color)
            if int(n) > max_map[color]:
                ok = False
                break
        if ok:
            ok_count += 1
    if ok_count == len(items):
        total+=id
print(total)

