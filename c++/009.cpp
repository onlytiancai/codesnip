#include <iostream>
#include <vector>
#include <map>
#include<algorithm>
using namespace std;

struct Goods 
{
 	string _name;
 	float _price;
};

class Person
{
public:
	Person() = default;

	Person(const char* name, int age = 0)//显示的写了构造函数，编译器就不会生成默认的构造函数
		:_name(name)
		, _age(age)
	{}

private:
	string _name;
	int _age;
};



int main()
{
	auto add = [](int a, int b)->int{return a + b; };	
	//[](int a, int b){return a + b;};
	cout << add(1, 2) << endl;

	Goods gds[] = { { "苹果", 2.1 }, { "香蕉", 3 }, { "橙子", 2.2 }, { "菠萝", 1.5 }, { "哈密瓜", 4 } };
	sort(gds, gds + sizeof(gds) / sizeof(gds[0]), 
	[](const Goods& left, const Goods& right)->bool{return left._price < right._price; });
	//[](const Goods& left, const Goods& right){return left._price < right._price;}
	for (auto& item : gds)
	{
		cout << item._name << ":" << item._price << endl;
	}

    int a = 10, b = 20;
	auto swap = [&a, &b]()mutable
	{
		int c = a;
		a = b;
		b = c;
	};
	swap();
	cout << "a:" << a << " " << "b:" << b << endl;
	return 0;

	Person s1("fl",10);//调用显示的拷贝构造
	Person s2;//调用default生成的默认拷贝构造
}

