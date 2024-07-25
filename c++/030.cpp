#include <iostream>
#include <string.h>
using namespace std;
class MyString {
public:
    MyString() : data(nullptr), len(0) {}

    MyString(const char* str) {
        len = strlen(str);
        data = new char[len + 1];
        strcpy(data, str);
    }

    MyString(const MyString& rhs) {
        len = rhs.len;
        data = new char[len + 1];
        strcpy(data, rhs.data);
    }

    MyString(MyString&& rhs) {
        len = rhs.len;	//将资源全部转到新的后，本身全部释放掉
        data = rhs.data;
        rhs.len = 0;
        rhs.data = nullptr;
    }

    ~MyString() {
        delete[] data;
    }

private:
    char* data;
    size_t len;
};

int main() {
    MyString str1("hello");
    MyString str2(std::move(str1));  // 使用std::move将左值转成右值引用
    return 0;
}
