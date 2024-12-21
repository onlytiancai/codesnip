import uuid
import os, sys

def generate_and_write_uuids(file_path):
    uuid_list = [str(uuid.uuid4()) for _ in range(1000000)]
    uuid_list.sort()
    with open(file_path, 'w') as file:
        for i, uuid_str in enumerate(uuid_list):
            if i % 10000 == 0:
                print('gen file', i)
            file.write(uuid_str + '\n')


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
            print(f'search: {i}, l={low}, m={mid},h={high}')

            f.seek(mid)
            if mid != 0: # 如果不正好在行首，则跳过当前行的一部分，保证从下一行开始
                f.readline()

            line = f.readline().strip()
            if not line:
                low = mid + 1
                continue

            if line < target:
                low = mid + 1
            elif line > target:
                high = mid - 1
            else:
                return True

    return False

file_path = "sorted_test.txt"
if not os.path.exists(file_path):
    print('gen file ...')
    generate_and_write_uuids(file_path)

target_string = sys.argv[1] 
exists = binary_search_file(file_path, target_string)

if exists:
    print(f"在 {file_path} 中找到了 {target_string}")
else:
    print(f"在 {file_path} 中未找到 {target_string}")
