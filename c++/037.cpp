#include <iostream>
#include <ranges>
#include <vector>

int main() {
  std::vector<int> vec = {1, 2, 3, 4, 5}; 

  // 用基于范围的for循环迭代
  for (int i : vec | std::views::filter([](int i){return i % 2 == 0;})) {
    std::cout << i << " "; 
  }

  // 输出:2 4
}

