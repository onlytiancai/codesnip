#include <iostream>

int main() {
    auto add = [](auto x, auto y) {
        return x + y;
    };
    
    std::cout << add(1, 2) << std::endl; // 输出 3
    std::cout << add(1.5, 2.5) << std::endl; // 输出 4
    
    return 0;
}
