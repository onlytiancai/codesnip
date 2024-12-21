import random
import os, sys

def generate_sorted_file_with_skips(filename, num_lines, max_skip=1000):
    current_number = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(1, num_lines + 1):
            skip = random.randint(1, max_skip)
            current_number += skip
            line = str(current_number) + '\n'
            f.write(line)
            
            # 每输出1万行打印一次进度
            if i % 10000 == 0:
                print(f"已生成 {i} 行")


def binary_search_file(filename, target):
    file_size = os.path.getsize(filename)
    print('文件大小：', file_size)
    with open(filename, 'r', encoding='utf-8') as f:
        low, high = 0, file_size
        i = 0 
        while low <= high:
            i = i + 1
            if i > 1000:
                print('检测到死循环，退出')
                break
            mid = (low + high) // 2
            #print(f'search: {i}, l={low}, m={mid},h={high}')

            f.seek(mid)
            if mid != 0: # 如果不正好在行首，则跳过当前行的一部分，保证从下一行开始
                f.readline()

            line = f.readline().strip()
            if not line:
                low = mid + 1
                continue
            # print('debug:', int(line), int(target))
            if int(line) < int(target):
                low = mid + 1
            elif int(line) > int(target):
                high = mid - 1
            else:
                return True

    return False

file_path = "sorted_skipped_numbers.txt"
if not os.path.exists(file_path):
    print('gen file ...')
    num_lines = 100000000  # 1亿行
    max_skip = 1000  # 每次最多跳过1000个数字
    generate_sorted_file_with_skips(file_path, num_lines, max_skip)


target_string = sys.argv[1] 
exists = binary_search_file(file_path, target_string)

if exists:
    print(f"在 {file_path} 中找到了 {target_string}")
else:
    print(f"在 {file_path} 中未找到 {target_string}")
