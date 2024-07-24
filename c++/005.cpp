#include <iostream>
#include <vector>
#include <map>
using namespace std;

// decltype的一些使用使用场景
template<class T1, class T2>
void F(T1 t1, T2 t2) 
{
	decltype(t1 * t2) ret;
	cout << typeid(ret).name() << endl;
}
int main()
{
	const int x = 1;
	double y = 2.2;
	decltype(x * y) ret;
	decltype(&y) p; 
	cout << typeid(ret).name() << endl; // ret的类型是double
	cout << typeid(p).name() << endl;  // p的类型是int*
	F(1, 'a');
    map<string, string> m = { { "insert", "插入" }, { "sort", "排序" } };
	auto it = m.begin();
	//vector<auto it> v;//错误
	vector<decltype(it)> v;//正确


	return 0;
}
