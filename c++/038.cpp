#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <algorithm>

using namespace std;



auto fill_map(ifstream& file) {
    auto p_map= make_unique<unordered_map<string, int>>(); 
    string word;
    while (file >> word) {
        if (!isalpha(word.front())) continue;
        for (auto &ch : word) {
            ch = tolower(ch);
        }
        ++(*p_map)[word];
    }
    return p_map;
}

auto fill_vec(unordered_map<string, int>& map) {
    auto p_vec = make_unique<vector<pair<string, int>>>(); 
    for (const auto &pair : map) {
        (*p_vec).push_back(pair);
    }
    return p_vec;
}

void sort_vec(vector<pair<string, int>>& vec) {
    sort(vec.begin(), vec.end(),
         [](const auto& a, const auto& b) {
            return a.second > b.second;
         });
}

void print_top_n_v0(int n, vector<pair<string, int>> vec) {
    for (int i = 0; i < 10 && i < vec.size(); ++i) {
        cout << vec[i].first << ": " << vec[i].second << endl;
    }
}
void print_top_n_v1(int n, vector<pair<string, int>>& vec) {
    for (int i = 0; i < 10 && i < vec.size(); ++i) {
        cout << vec[i].first << ": " << vec[i].second << endl;
    }
}
void print_top_n_v2(int n, vector<pair<string, int>>&& vec) {
    for (int i = 0; i < 10 && i < vec.size(); ++i) {
        cout << vec[i].first << ": " << vec[i].second << endl;
    }
}
void print_top_n_v3(int n, unique_ptr<vector<pair<string, int>>> p_vec) {
    auto vec = *p_vec;
    for (int i = 0; i < 10 && i < vec.size(); ++i) {
        cout << vec[i].first << ": " << vec[i].second << endl;
    }
}

int main() {
    ifstream file("../txt/scan.txt");
    if (!file) {
        cerr << "无法打开文件" << endl;
        return 1;
    }
 
    auto p_map = fill_map(file);
    auto p_vec = fill_vec(*p_map);
    sort_vec(*p_vec);
    print_top_n_v0(10, *p_vec);
    print_top_n_v1(10, *p_vec);
    print_top_n_v2(10, move(*p_vec));
    print_top_n_v3(10, move(p_vec));
    return 0;
}
