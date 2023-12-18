from collections import defaultdict
seeds = []
map = defaultdict(list)
map_name = ''
for i, line in enumerate(open('data5.txt')):
    line = line.strip()
    print(i, line)
    if not line:
        continue
    if i == 0:
        seeds = [int(x.strip()) for x in line.split(':')[1].split(' ') if x]
        print('seeds:', seeds)
        continue
    if line.endswith('map:'):
        map_name = line.split(' ')[0]
        continue
    map[map_name].append([int(x) for x in line.split(' ')])
print(map)

def trans(data, map_name):
    for x in data:
        result = -1
        for dst,src,len in map[map_name]:
            if x >= src and x < src + len:
                result = dst + (x - src)
                break
        if result == -1:
            result = x 
        yield result

data = seeds
for name in ['seed-to-soil', 'soil-to-fertilizer', 'fertilizer-to-water', 'water-to-light', 
        'light-to-temperature', 'temperature-to-humidity', 'humidity-to-location']:
    results = list(trans(data, name))
    data = results
print(min(data), data)

