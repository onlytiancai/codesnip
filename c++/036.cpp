#include <iostream>
#include <optional>

std::optional<int> divide(int a, int b) {
    if (b == 0) {
        return std::nullopt; // 表示不存在值
    } else {
        return a / b;
    }
}

int main() {
    auto result = divide(10, 2);
    if (result) {
        std::cout << "Result: " << *result << std::endl;
    } else {
        std::cout << "Error: Division by zero" << std::endl;
    }

    result = divide(10, 0);
    if (result) {
        std::cout << "Result: " << *result << std::endl;
    } else {
        std::cout << "Error: Division by zero" << std::endl;
    }

    return 0;
}
