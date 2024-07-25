#include <iostream>
constexpr int factorial(int n) {
    return n <= 1 ? 1 : n * factorial(n - 1);
}

int main() {
    constexpr int n = 5;
    int arr[factorial(n)]; // 使用常量表达式计算数组大小

    int a = 0b101010;
    std::cout << a << std::endl; //输出42

    auto arr2 = {1, 2, 3, 4};
    return 0;
}
