def trans(src, dst, pair):
    results, remain=[],[]
    print(f'pair:{pair}, src:{src}, dst:{dst}')
    if pair[0] < src[0] and pair[1] > pair[0]:
        print('111', pair)
        remain.append([pair[0], min(pair[1], src[0]-1)])
        pair[0] = min(pair[1], src[0]-1)+1
    if pair[0] >= src[0] and pair[0] <= src[1] and pair[1] > pair[0]:
        begin, end = max(pair[0],src[0]), min(pair[1], src[1])
        print('222', pair, begin, end, [dst[0]+begin-src[0], dst[0]+end-src[0]])
        results.append([dst[0]+begin-src[0], dst[0]+end-src[0]])
        pair[0] = min(pair[1], src[1])+1
    if pair[1] > pair[0]:
        print('333', pair)
        remain.append(pair)
    return results, remain 


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
    next_seeds = []
   
    print('process seeds:', seeds)
    remain = []
    for seed in seeds:
        print('000:', seed)
        for m in map:
            next_remain = []
            print('process remain:', remain)
            for x  in remain:
                results, new_remain = trans(m[0], m[1], x)
                next_seeds.extend(results)
                next_remain.extend(new_remain)

            print('process seed:', seed)
            results, new_remain = trans(m[0], m[1], seed)
            next_seeds.extend(results)
            next_remain.extend(new_remain)
            remain = next_remain

    next_seeds.extend(remain)
    print('next_seeds:', next_seeds)
    print()
    seeds = next_seeds 


print(seeds)
print('next', min(seeds))
