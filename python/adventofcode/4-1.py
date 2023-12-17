lines = [line.strip() for line in open('data4.txt')]
total = 0
for i, line in enumerate(lines):
    arr = line.split('|')
    win_nums = [x for x in arr[0].split(':')[1].split(' ') if x]
    my_nums = [x for x in arr[1].split(' ') if x]
    print(i, win_nums, my_nums)
    current = 0 
    for n in my_nums:
        if n in win_nums:
            if current == 0:
                current = 1
            else:
                current *= 2
    print('current', current)
    total += current
print(total)
