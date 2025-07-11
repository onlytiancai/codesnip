import time
import random

MAX_UINT64 = 2**64 - 1
TOTAL_RANGE = MAX_UINT64 + 1  # 即 2^64
DEFAULT_STEP = TOTAL_RANGE // 100  # 万分之一范围
MIN_BLOCK_SIZE = 1000  # 最小可再切分块

def do_task(start, end):
    """
    模拟处理某个 cityHash64 范围，可以替换为你自己的逻辑。
    抛出异常代表失败。
    """
    percent = start / TOTAL_RANGE * 100
    print(f"🟢 {percent:.2f}% {end-start}| Processing range: {start} ~ {end}")

    if random.random() < 0.5:
        print(f"Processing ERROR: {start} ~ {end}")
        raise RuntimeError("模拟随机失败")
    time.sleep(0.5)
    print(f"✅ Success: {start} ~ {end}")

def process_range(start, end):
    try:
        do_task(start, end)
    except Exception as e:
        if end - start <= MIN_BLOCK_SIZE:
            print(f"⛔ Too small to split: {start} ~ {end}. Skip. Error: {e}")
            return
        mid = (start + end) // 2
        process_range(start, mid)
        process_range(mid + 1, end)

def main():
    step = DEFAULT_STEP
    current = 0
    while current <= MAX_UINT64:
        end = min(current + step - 1, MAX_UINT64)
        process_range(current, end)
        current = end + 1

if __name__ == "__main__":
    main()
