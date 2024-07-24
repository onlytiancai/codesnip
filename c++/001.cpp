#include <iostream>
 
int main() {
    int sum = 0;
    for (int i = 2; i <= 100; i += 2) {
        sum += i;
    }
    std::cout << "偶数和为: " << sum << std::endl;
    return 0;
}
