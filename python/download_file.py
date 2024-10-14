# download_file.py
import time
import sys

for i in range(101):
    print(f"\r进度: {i}%", end='', flush=True)
    time.sleep(0.1)  # 模拟下载延迟
print()  # 换行
