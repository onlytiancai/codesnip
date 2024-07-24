#include <iostream>
#include <vector>
#include <map>
#include<algorithm>
#include <functional>

using namespace std;


int Plus(int a, int b)
{
	return a + b;
}

class Sub
{
public:
	int sub(int a, int b)
	{
		return a - b;			
	}
};
int main()
{
	//表示绑定函数plus 参数分别由调用 func1 的第一，二个参数指定
	std::function<int(int, int)> func1 = std::bind(Plus, placeholders::_1, placeholders::_2);
	//auto func1 = std::bind(Plus, placeholders::_1, placeholders::_2);

	//func2的类型为 function<void(int, int, int)> 与func1类型一样
	//表示绑定函数plus,第一个参数为1，第二个参数为2
	auto func2 = std::bind(Plus, 1, 2);
	cout << func1(1, 2) << endl;
	cout << func2() << endl;

	//绑定成员函数
	Sub s;
	std::function<int(int, int)> func3 = std::bind(&Sub::sub, s, placeholders::_1, placeholders::_2);

	// 参数调换顺序
	std::function<int(int, int)> func4 = std::bind(&Sub::sub, s, placeholders::_2, placeholders::_1);
	cout << func3(1, 2) << endl;
	cout << func4(1, 2) << endl;
	return 0; 
}

