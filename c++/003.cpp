#include <iostream>
using namespace std;

struct Point
{
 	int _x;
 	int _y;
};
int main()
{
 	int array1[] = { 1, 2, 3, 4, 5 };
 	int array2[5] = { 0 };
 	Point p = { 1, 2 };
    // C++11中列表初始化也可以适用于new表达式中
 	int* pa = new int[4]{ 0 };

    initializer_list<string> lt = { "hello", "feng", "lei" };
	for (auto item : lt)
	{
		cout << item << " ";
	}
	cout << endl;

	return 0; 
}
