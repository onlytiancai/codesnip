import re
lines = [line.strip() for line in open('data3.txt')]
len_lines = len(lines)
total = 0
for i, line in enumerate(lines):
    prev_line, next_line = '', ''
    numbers = list(re.finditer('\d+', line))
    print(line, numbers)
    if i > 0:
        prev_line = lines[i-1]
    if i < len_lines -1:
        next_line = lines[i+1]
    for num in numbers:
        print(111, num)
        left, right = num.span()
        left -= 1
        right += 1
        if left < 0:
            left = 0
        if right > len(line):
            right = len(line)
        found = False
        for line2 in [prev_line, line, next_line]:
            temp = line2[left:right]
            print('\t', temp, left, right)
            if re.findall('[^\d.]', temp):
                found = True
                break
        if found:
            print(222, num.group())
            total += int(num.group())
print(total)
