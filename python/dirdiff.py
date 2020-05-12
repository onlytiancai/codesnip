import os
src = 'C:\\'
dst = r'E:\action-pc-bakup\github'
exclude = '.git'

total_file = 0
diff_files = set()

for root, dirs, files in os.walk(src, topdown=False):
    if root.find(exclude) != -1:
        continue 

    dirpath = root[len(src):]
    for name in files:
        total_file += 1
        if total_file % 1000 == 0:
            print("%s file scaned" % total_file)

        src_file = os.path.join(root, name)
        dst_file = os.path.join(dst + dirpath, name)

        src_stats = os.stat(src_file) if os.path.exists(src_file) else os.stat_result(0 for i in range(10))
        dst_stats = os.stat(dst_file) if os.path.exists(dst_file) else os.stat_result(0 for i in range(10))

        mtime_diff = abs(src_stats.st_mtime - dst_stats.st_mtime) > 1
        size_diff = src_stats.st_size != dst_stats.st_size
        if any([mtime_diff, size_diff]):
            diff_files.add(src_file)

print('total files:%s, diff files:%s ' % (total_file, len(diff_files)))