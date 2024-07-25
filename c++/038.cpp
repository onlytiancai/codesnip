#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <vector>
#include <algorithm>
 
int main() {
    std::ifstream file("../txt/scan.txt"); // 替换为你的文本文件路径
    std::string word;
    std::map<std::string, int> word_count;
    std::vector<std::pair<std::string, int>> word_count_vec;
 
    // 检查文件是否成功打开
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 1;
    }
 
    // 读取文件内容并统计每个单词出现的次数
    while (file >> word) {
        // 忽略不是字母的单词
        if (!std::isalpha(word.front())) continue;
        for (char &ch : word) {
            ch = std::tolower(ch); // 转换为小写
        }
        ++word_count[word];
    }
 
    // 将map中的数据复制到vector中
    for (const auto &pair : word_count) {
        word_count_vec.push_back(pair);
    }
 
    // 根据单词出现次数进行排序
    std::sort(word_count_vec.begin(), word_count_vec.end(),
              [](const std::pair<std::string, int> &a, const std::pair<std::string, int> &b) {
                  return a.second > b.second;
              });
 
    // 输出出现次数最多的10个单词
    for (int i = 0; i < 10 && i < word_count_vec.size(); ++i) {
        std::cout << word_count_vec[i].first << ": " << word_count_vec[i].second << std::endl;
    }
 
    return 0;
}
