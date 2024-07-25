#include <iostream>
#include <sstream>
#include <vector>
 
std::vector<std::string> splitStringBySpace(const std::string &str) {
    std::istringstream iss(str); // 创建字符串流
    std::vector<std::string> tokens;
    std::string token;
 
    while (iss >> token) { // 读取每个由空格分隔的单词
        tokens.push_back(token);
    }
 
    return tokens;
}
 
int main() {
    std::string str = "这 是 一 个 测 试 字 符 串";
    std::vector<std::string> result = splitStringBySpace(str);
 
    for (const auto &word : result) {
        std::cout << word << std::endl;
    }
 
    return 0;
}
