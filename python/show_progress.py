# main.py
import subprocess

# 调用下载文件脚本
process = subprocess.Popen(
    ['/usr/bin/python3', './download_file.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# 逐行读取输出
while True:
    output = process.stdout.readline()
    if output == '' and process.poll() is not None:
        break
    if output:
        print(f"读取输出: {output.strip()}")  # 打印输出

# 等待子进程结束
process.wait()
