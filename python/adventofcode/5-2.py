def trans(map, a):
    result=[]
    print('seed', a)
    for k,v in sorted(map):
        print(k,v)
        if a[0] < k[0] and a[0]<a[1]:
            print('out', a[0], min(a[1], k[0]-1))
            result.append([a[0], min(a[1], k[0]-1)])
            a[0] = min(a[1],k[0]-1)+1

        if a[1] >= k[0] and a[1] <= k[1] and a[0]<a[1]:
            print('in', max(a[0], k[0]), min(a[1], k[1]))
            result.append([v[0]+max(a[0], k[0])-k[0],v[0]+min(a[1], k[1])-k[0]])
            a[0]=min(a[1], k[1])+1

    if a[0]<a[1]:
        print('fin', a)
        result.append([a[0], a[1]])
    print(result)
    return result

seeds = []
maps = []
map = []
map_name = ''
for i, line in enumerate(open('data5.txt')):
    line = line.strip()
    if not line:
        continue
    if i == 0:
        a = [int(x.strip()) for x in line.split(':')[1].split(' ') if x]
        seeds = [[a[i],a[i]+a[i+1]] for i in range(0, len(a), 2)]
        continue
    if line.endswith('map:'):
        if map:
            maps.append(map)
            map=[]
        map_name = line.split(' ')[0]
        continue
    dst,src,rng = [int(x) for x in line.split(' ')]
    map.append(((src,src+rng),(dst, dst+rng)))

if map:
    maps.append(map)
print('seeds:', seeds)

for map in maps:
    print('map:', map)
    results = []
    for seed in seeds:
        results.extend(trans(map, seed))
    print('results:',results)
    seeds = results

print(seeds)
print(min(seeds))
