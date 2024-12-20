import heapq, csv, sys
from collections import defaultdict
from itertools import product

def output_merge_cache(writer, line, values):
    for arr in product(*values):
        arr = list(arr)
        arr.insert(0, line)
        writer.writerow(arr)

def find_common_lines(outfile, files):
    writer = csv.writer(outfile)
    writer.writerow(['line']+[f'{name}_line_no' for name in files])

    file_iters = [(((line.strip(),i+1) for i,line in enumerate(open(f))), f) for f in files]
    merged = heapq.merge(*[(lambda iter, name: ((name, i, line) for line, i in iter))(*file_iter)
                          for file_iter in file_iters], key=lambda x: x[2])
    last_line = ''
    merge_cache = defaultdict(list)
    for filename, line_no, line in merged:
        # print(filename, line_no, line)
        if line != last_line:
            if len(merge_cache) == len(files):
                output_merge_cache(writer, last_line, merge_cache.values())
            merge_cache.clear()

        last_line = line
        merge_cache[filename].append(line_no)

find_common_lines(sys.stdout, ['file1.txt', 'file2.txt', 'file3.txt'])
