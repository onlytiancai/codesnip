// 第 11 步：return 和递归
fn factorial(n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
print(factorial(5));
