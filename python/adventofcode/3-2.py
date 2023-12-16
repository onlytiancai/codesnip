import re
lines = [line.strip() for line in open('data3.txt')]
total, line_count, max_line = 0, len(lines), len(lines[0])
empty_line = ' ' * max_line
for i, line in enumerate(lines):
    for j, ch in enumerate(line):
        if ch == '*':
            print(i,j,ch)
            prev_line = lines[i-1] if i > 0 else empty_line 
            next_line = lines[i+1] if i < line_count -1 else empty_line 
            nums = []
            for temp in [prev_line, line, next_line]:
                for m in re.finditer('\d+', temp):
                    if m.start() <= j -1 < m.end() or m.start() <= j < m.end() or m.start() <= j +1 < m.end():
                        nums.append(int(m.group()))
            if len(nums) == 2:
                total += nums[0]*nums[1]
print(total)
