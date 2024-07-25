#include <memory>
#include <iostream>

class MyClass {
public:
    MyClass(int value) : m_value(value) {
        std::cout << "MyClass constructor called with value: " << m_value << std::endl;
    }
    ~MyClass() {
        std::cout << "MyClass destructor called with value: " << m_value << std::endl;
    }
private:
    int m_value;
};

int main() {
    auto ptr = std::make_unique<MyClass>(42);
    return 0;
}
