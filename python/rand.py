def linear_congruential_generator(seed, a, c, m, num_values):
    random_numbers = []
    X = seed
    for _ in range(num_values):
        X = (a * X + c) % m
        random_numbers.append(X)
    return random_numbers

# 示例参数
seed = 5  # 初始种子
a = 1664525  # 乘数
c = 1013904223  # 增量
m = 2**32  # 模数
num_values = 10  # 生成随机数的数量

# 生成随机数
random_numbers = linear_congruential_generator(seed, a, c, m, num_values)
print(random_numbers)
