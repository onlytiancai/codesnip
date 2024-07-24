#include <iostream>
#include <vector>
#include <map>
#include<algorithm>
#include <functional>
#include <thread>

using namespace std;


void ThreadFunc1(int& x) 
{
	x += 10;
}
void ThreadFunc2(int* x) 
{
	*x += 10;
}
int main()
{
	int a = 10;

	// 如果想要通过形参改变外部实参时，必须借助std::ref()函数
	thread t2(ThreadFunc1, std::ref(a));
	t2.join();
	cout << a << endl;

	// 地址的拷贝
	thread t3(ThreadFunc2, &a);
	t3.join();
	cout << a << endl;
	return 0;
}
