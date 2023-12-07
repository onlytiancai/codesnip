total = 0
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
    print('line=[%s],arr=%s,first=[%s],last=[%s]' % (line, arr, first, last))
print(total)
