#include <iostream>
#include <vector>
#include <map>
#include <cmath>
using namespace std;

int main()
{
	// 以下的p、b、c、*p都是左值，都能被取地址
	int* p = new int(0);
	int b = 1;
	const int c = 2;
	// 以下几个是对上面左值的左值引用
	int*& rp = p;
	int& rb = b;
	const int& rc = c;
	int& pvalue = *p;

    double x = 1.1, y = 2.2;
	// 以下几个都是常见的右值
	10;
	x + y;
	fmin(x, y);
	// 以下几个都是对右值的右值引用
	int&& rr1 = 10;
	double&& rr2 = x + y;
	double&& rr3 = fmin(x, y);
	// 以下编译会报错：error C2106: “=”: 左操作数必须为左值
	//10 = 1;
	//x + y = 1;
	//fmin(x, y) = 1;
   
    //引用前必须加上const， 否则会报错
	const int& r = 10;
	const double& r1 = x + y;
	const double& r2 = fmin(x, y);

	
	//右值引用引用左值
	int* && rr4 = move(p);
	int && rr5 = move(*p);
	int && rr6 = move(b);
	const int && rr7 = move(c);

	return 0;
}

