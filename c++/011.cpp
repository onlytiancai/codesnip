#include <iostream>
#include <vector>
#include <map>
#include<algorithm>
using namespace std;

//也可以写成这样
template <class T>
int PrintArg(T val)
{
	cout << typeid(T).name() << ":" << val;
	return 0;
}
//展开函数
template <class ...Args>
void ShowList(Args... args)
{
	int arr[] = { PrintArg(args)... };
	cout << endl;
}

int main()
{
	ShowList(1);
	ShowList(1, 'A');
	ShowList(1, 'A', string("fl"));
	return 0;
}
