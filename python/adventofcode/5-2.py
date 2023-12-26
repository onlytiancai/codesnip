def trans(src, dst, pair):
    results, remain=[],[]
    if pair[0] < src[0] and pair[1] > pair[0]:
        remain.append([pair[0], min(pair[1], src[0]-1)])
        pair[0] = min(pair[1], src[0]-1)+1
    if pair[0] >= src[0] and pair[0] <= src[1] and pair[1] > pair[0]:
        begin, end = max(pair[0],src[0]), min(pair[1], src[1])
        results.append([dst[0]+begin-src[0], dst[0]+end-src[0]])
        pair[0] = min(pair[1], src[1])+1
    if pair[1] > pair[0]:
        remain.append(pair)
    return results, remain 

def process_map(seeds, map):
    next_seeds = []
    remain = []
    for seed in seeds:
        for m in map:
            next_remain = []
            for x  in remain:
                results, new_remain = trans(m[0], m[1], x)
                next_seeds.extend(results)
                next_remain.extend(new_remain)
            results, new_remain = trans(m[0], m[1], seed)
            next_seeds.extend(results)
            next_remain.extend(new_remain)
            remain = next_remain
    next_seeds.extend(remain)
    print('next_seeds:', next_seeds)
    return next_seeds 

lines = [x.strip() for x in open('data5.txt').readlines() if x.strip()]
seeds,map = [],[]
for i, line in enumerate(lines):
    if i == 0:
        a = [int(x.strip()) for x in line.split(':')[1].split(' ') if x]
        seeds = [[a[i],a[i]+a[i+1]] for i in range(0, len(a), 2)]
        continue
    if i == len(lines)-1 or line.endswith('map:'):
        if map:
            seeds,map = process_map(seeds, map),[]
        continue
    dst,src,rng = [int(x) for x in line.split(' ')]
    map.append(((src,src+rng),(dst, dst+rng)))
print('result', min(seeds))
