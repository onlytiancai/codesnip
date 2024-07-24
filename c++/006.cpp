#include <iostream>
#include <vector>
#include <map>
using namespace std;


int main()
{
	vector<char> v = { 'h', 'e', 'l', 'l', 'o' };
	for (vector<char>::iterator it = v.begin(); it != v.end(); ++it)
	{
		cout << *(it) << " ";
	}
	cout << endl;
    for (auto it : v)
	{
		cout << it << " ";
	}
	cout << endl;
}
