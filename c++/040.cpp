#include <memory>
#include <unordered_map>
#include <iostream>

using namespace std;

int main(int argc, char *argv[])
{
    auto p_word_count = make_unique<unordered_map<string, int>>(); 
    auto word_count = *p_word_count;
    word_count["a"] = 1;
    cout << word_count["a"]<<endl;
    return 0;
}
