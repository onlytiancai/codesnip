#include <iostream>
#include <vector>
#include <map>
#include <cmath>
using namespace std;


void Fun(int &x)
{ 
	cout << "左值引用" << endl; 
}
void Fun(const int &x)
{ 
	cout << "const 左值引用" << endl; 
}
void Fun(int &&x)
{ 
	cout << "右值引用" << endl;
}
void Fun(const int &&x)
{ 
	cout << "const 右值引用" << endl;

}
template<typename T>
void PerfectForward(T&& t) 
{
	//Fun(t);
    Fun(std::forward<T>(t));
}
int main()
{
	PerfectForward(10);           // 右值
	int a;
	PerfectForward(a);            // 左值
	PerfectForward(std::move(a)); // 右值
	const int b = 8;
	PerfectForward(b);            // const 左值
	PerfectForward(std::move(b)); // const 右值
	return 0;
}
