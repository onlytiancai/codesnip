import os
import random
import itertools


def analyze_sorted_file_with_groupby_stream_and_max_count(filename):
    # 打开文件并逐行读取
    with open(filename, 'r', encoding='utf-8') as f:
        # 逐行读取数字并将其转换为整数
        numbers = (int(line.strip()) for line in f)
        
        total_sum = 0
        max_count = 0
        max_number = None
        print("每个数字出现的次数：")
        
        # 使用 itertools.groupby 对相邻相同的数字进行分组
        for num, group in itertools.groupby(numbers):
            count = len(list(group))  # 统计当前数字出现的次数
            total_sum += num * count  # 累加到总和
            
            # 检查当前数字的出现次数是否是最多的
            if count > max_count:
                max_count = count
                max_number = num
                
            print(f"数字 {num} 出现 {count} 次，当前出现次数最多的数字是 {max_number}，出现 {max_count} 次")
        
        # 打印数字的总和
        print(f"\n数字的总和是: {total_sum}")
        
        # 打印出现次数最多的数字
        print(f"\n出现次数最多的数字是: {max_number}，出现次数为: {max_count}")

def generate_sorted_file_with_skips_and_duplicates(filename, num_lines, max_skip=1000, max_repeat=5):
    current_number = 0
    with open(filename, 'w', encoding='utf-8') as f:
        for i in range(1, num_lines + 1):
            # 随机决定跳过的数字数量
            skip = random.randint(1, max_skip)
            current_number += skip
            
            # 随机决定该数字是否重复以及重复次数
            repeat_count = random.randint(1, max_repeat)  # 随机重复次数
            
            # 写入重复的数字
            for _ in range(repeat_count):
                f.write(str(current_number) + '\n')
            
            # 每输出1万行打印一次进度
            if i % 10000 == 0:
                print(f"已生成 {i} 行")
    
# 使用示例
filename = 'sorted_skipped_numbers_with_duplicates.txt'
if not os.path.exists(filename):
    num_lines = 1000000
    max_skip = 1000
    max_repeat = 100
    generate_sorted_file_with_skips_and_duplicates(filename, num_lines, max_skip, max_repeat)
    print(f"文件 {filename} 已生成，包含 {num_lines} 行排序文本，且包含随机重复的数字。")

analyze_sorted_file_with_groupby_stream_and_max_count(filename)
