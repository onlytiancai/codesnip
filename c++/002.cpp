#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    std::vector<int> vec4{1,2,3,4};
    for (int num : vec4) {
        std::cout << num << " ";
    }
    return 0;
}
