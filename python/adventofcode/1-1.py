import re
total = 0
for line in open('data.txt'):
    line = line.strip()
    arr = re.findall('\d',line)
    total += int(arr[0]+arr[-1])
    print(line, arr, int(arr[0]+arr[-1]), total)
print(total)
