total = 0
import re
map = {
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
}
    
for line in open('data.txt'):
    arr = []
    line = line.strip().lower()
    arr2 = re.findall('\d|one|two|three|four|five|six|seven|eight|nine', line)
    current2 = int(map.get(arr2[0], arr2[0])+map.get(arr2[-1], arr2[-1]))
    for temp in [line[i:] for i in range(len(line))]:
        if temp[0].isnumeric():
            arr.append(temp[0])
        else:
            for k in map:
                if temp.startswith(k):
                    arr.append(map[k])
        

    first = arr[0] 
    last = arr[-1] 
    current = int(str(first)+str(last))
    total += current 
    if current2 != current:
        print('line=[%s],arr=%s,arr2=%s' % (line, arr, arr2))
print(total)
