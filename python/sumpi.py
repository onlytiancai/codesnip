import decimal

def calculate_pig_sum():
    # 设置足够的精度，以便获取小数点后1000位的pi值
    decimal.getcontext().prec = 1005  # 略高于所需的小数位数，以避免四舍五入错误

    pi = decimal.Decimal(3)
    n = 6

    while True:
        term = (decimal.Decimal(1) / (2**n * (2**n - 1))) * (-1)**(n+1)
        pi += term
        if abs(term) < decimal.Decimal(10)**(-1000):
            break
        n += 1

    # 转换pi为字符串，去除小数点前的整数部分
    pi_str = str(pi)[2:]
    print('pi=', pi)

    # 前面的数字可能不足1000位，截取实际的小数位数
    pi_digits = list(pi_str[:1000])

    total = 0
    for d in pi_digits:
        total += int(d)

    print(f"圆周率π的小数点后1000位数字之和为：{total}")

calculate_pig_sum()
