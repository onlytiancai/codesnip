#include <iostream>
#include <vector>
#include <map>
#include<algorithm>
using namespace std;


void showlist()
{
	cout << "end" << endl;
}

//解析并打印参数包中每个参数的类型及值
//传三个参数,val获取一个，剩下的在参数包args中，递归下去，知道参数为0为止，最后将调用showlist()
template<class T, class ...Args>
void showlist(T val, Args... args)
{
	cout << typeid(val).name() << ":" << val << endl;
	showlist(args...);
}

template <class T>
void PrintArg(T val)
{
	cout << typeid(T).name() << ":" << val;
}
//展开函数
//这种展开参数包的方式，不需要通过递归终止函数，是直接在expand函数体中展开的, printarg
//不是一个递归终止函数，只是一个处理参数包中每一个参数的函数。这种就地展开参数包的方式
//实现的关键是逗号表达式。我们知道逗号表达式会按顺序执行逗号前面的表达式，
//expand函数中的逗号表达式：(printarg(args), 0)，也是按照这个执行顺序，先执行
//printarg(args)，再得到逗号表达式的结果0
template <class ...Args>
void ShowList(Args... args)
{
	int arr[] = { (PrintArg(args), 0)... };
	cout << endl;
}

int main()
{
	showlist("fl", 2, string("hehe"));

	ShowList(1);
	ShowList(1, 'A');
	ShowList(1, 'A', string("fl"));
	return 0;
}

