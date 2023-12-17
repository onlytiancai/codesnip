from collections import defaultdict
stats = defaultdict(int)
for i, line in enumerate(open('data4.txt')):
    arr = line.strip().split('|')
    win_nums = [x for x in arr[0].split(':')[1].split(' ') if x]
    my_nums = [x for x in arr[1].split(' ') if x]
    stats[i] += 1
    current = len(list(n for n in my_nums if n in win_nums))
    for j in range(i+1, i+current+1):
        stats[j] += stats[i] 
print(sum(stats.values()))
