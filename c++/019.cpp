#include <iostream>
using namespace std;
class Test
{
public:
    void TestWork(int index)
    {
        std::cout << "TestWork 1" << std::endl;
    }
    void TestWork(int * index)
    {
        std::cout << "TestWork 2" << std::endl;
    }
};

template<typename T, typename U>
auto add(T x, U y) -> decltype(x+y) {
    return x+y;
}


int main()
{
    Test test;
    test.TestWork(nullptr);	//输出 TestWork 2
    cout << "add result," << add(5,3.3) << endl;
}
