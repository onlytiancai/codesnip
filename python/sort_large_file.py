import os
import heapq
import uuid
from multiprocessing import Pool

def print_memory_usage():
    """打印当前进程的内存使用情况"""
    with open(f"/proc/{os.getpid()}/status", 'r') as f:
        for line in f:
            if line.startswith("VmRSS"):
                memory_usage = int(line.split()[1]) / 1024  # 转换为 MB
                print(f"Current memory usage: {memory_usage:.2f} MB")
                break

def sort_and_save(chunk_file):
    """对单个分块文件进行排序并保存"""
    print(f"Sorting chunk file: {chunk_file}")
    print_memory_usage()
    with open(chunk_file, 'r') as f:
        lines = f.readlines()
    lines.sort()  # 在内存中排序
    sorted_chunk_file = chunk_file + '_sorted'
    with open(sorted_chunk_file, 'w') as f:
        f.writelines(lines)
    print(f"Finished sorting and saving: {sorted_chunk_file}")
    return sorted_chunk_file

def merge_sorted_files(sorted_files, output_file):
    """归并多个排序后的文件"""
    print(f"Starting merge of {len(sorted_files)} sorted files into {output_file}")
    print_memory_usage()
    with open(output_file, 'w') as outfile:
        # 创建文件生成器以节省内存
        files = [open(file, 'r') for file in sorted_files]
        min_heap = [(file.readline().strip(), i) for i, file in enumerate(files)]
        heapq.heapify(min_heap)

        line_count = 0
        while min_heap:
            smallest, file_idx = heapq.heappop(min_heap)
            outfile.write(smallest + '\n')
            next_line = files[file_idx].readline().strip()
            if next_line:
                heapq.heappush(min_heap, (next_line, file_idx))

            line_count += 1
            if line_count % 100000 == 0:
                print(f"Merged {line_count} lines so far")
                print_memory_usage()

        for file in files:
            file.close()
    print(f"Finished merging files into {output_file}")

def split_file(input_file, chunk_size):
    """将大文件分块"""
    print(f"Splitting file: {input_file} into chunks of size {chunk_size} bytes")
    print_memory_usage()
    chunks = []
    with open(input_file, 'r') as f:
        chunk = []
        chunk_count = 0
        line_count = 0
        for line in f:
            chunk.append(line)
            line_count += 1
            if len(chunk) * len(line) >= chunk_size:
                chunk_file = f'{input_file}_chunk_{chunk_count}'
                with open(chunk_file, 'w') as chunk_f:
                    chunk_f.writelines(chunk)
                chunks.append(chunk_file)
                print(f"Created chunk file: {chunk_file}")
                print_memory_usage()
                chunk_count += 1
                chunk = []

            if line_count % 100000 == 0:
                print(f"Processed {line_count} lines so far")
                print_memory_usage()

        # 写入最后一块
        if chunk:
            chunk_file = f'{input_file}_chunk_{chunk_count}'
            with open(chunk_file, 'w') as chunk_f:
                chunk_f.writelines(chunk)
            chunks.append(chunk_file)
            print(f"Created chunk file: {chunk_file}")

    print(f"Finished splitting file into {len(chunks)} chunks")
    return chunks

def external_sort(input_file, output_file, chunk_size=1024 * 1024 * 100):  # 默认100MB块
    """外部排序主函数"""
    # Step 1: 分块文件
    chunks = split_file(input_file, chunk_size)

    # Step 2: 多进程排序每块文件
    print(f"Starting parallel sorting of {len(chunks)} chunks")
    with Pool(processes=os.cpu_count()) as pool:
        sorted_files = pool.map(sort_and_save, chunks)
    print(f"Finished sorting all chunks")

    # Step 3: 归并所有排序后的文件
    merge_sorted_files(sorted_files, output_file)

    # 清理临时文件
    print(f"Cleaning up temporary files")
    for file in chunks + sorted_files:
        os.remove(file)
    print(f"Finished cleanup")

def generate_large_file(file_path, num_lines):
    """生成一个大文件，每行是一个随机的UUID"""
    print(f"Generating large file: {file_path} with {num_lines} lines")
    print_memory_usage()
    with open(file_path, 'w') as f:
        for i in range(num_lines):
            f.write(str(uuid.uuid4()) + '\n')
            if (i + 1) % 100000 == 0:
                print(f"Generated {i + 1} lines so far")
                print_memory_usage()
    print(f"Finished generating file: {file_path}")

# 示例用法
if __name__ == '__main__':
    input_file = 'all_large.txt'
    output_file = 'sorted_output_file.txt'

    # 生成一个大文件
    # generate_large_file(input_file, 10**7)  # 生成包含一千万行的文件

    # 对大文件进行排序
    external_sort(input_file, output_file)
