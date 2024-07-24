#include <iostream>
#include <string.h>
#include <map>
using namespace std;

int main()
{
	int i = 10;
	auto p = &i;
	auto pf = strcpy;
	cout << typeid(p).name() << endl;
	cout << typeid(pf).name() << endl;
	map<string, string> dict = { { "sort", "排序" }, { "insert", "插入" } };
	//map<string, string>::iterator it = dict.begin();
	auto it = dict.begin();//等价于上面的写法
	return 0;
}
