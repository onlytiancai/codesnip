#include <iostream>
#include <utility>
#include <vector>

struct Point {
    int x;
    int y;
};


int main()
{
    int x = 1;
    int y = std::exchange(x, 2);
    std::cout << "x = " << x << ", y = " << y << std::endl; //输出x = 2, y = 1

    Point p{1, 2};
    auto [a, b] = p;

    std::vector<std::pair<int, std::string>> v{{1, "one"}, {2, "two"}, {3, "three"}};
    for (auto [key, value] : v) {
        std::cout << "key: " << key << ", value: " << value << std::endl;
    }

    constexpr auto square = [](auto x) { return x * x; };
    constexpr int result = square(5);
    std::cout << result << std::endl;

    return 0;
}
