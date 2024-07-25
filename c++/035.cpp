#include <iostream>
#include <string>
template<typename T>
void print(const T& t)
{
    if constexpr (std::is_integral_v<T>)
    {
        std::cout << "Integral type: " << t << std::endl;
    }
    else if constexpr (std::is_floating_point_v<T>)
    {
        std::cout << "Floating point type: " << t << std::endl;
    }
    else
    {
        std::cout << "Unknown type: " << t << std::endl;
    }
}

int main()
{
    int i = 42;
    float f = 3.14f;
    std::string s = "hello";

    print(i);   // 输出 "Integral type: 42"
    print(f);   // 输出 "Floating point type: 3.14"
    print(s);   // 输出 "Unknown type: hello"

    return 0;
}
