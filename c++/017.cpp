#include <iostream>
#include <thread>
#include<atomic>
using namespace std;
int main()
{
	atomic<int> sum(0);
	cout << "Before joining,sum = " << sum << std::endl;
	thread t1([&]{
		for (int i = 0; i < 100000; ++i)
		{
			++sum;
		}
	});
	thread t2([&]{
		for (int i = 0; i < 100000; ++i)
		{
			++sum;
		}
	});

	t1.join();
	t2.join();
	cout << "After joining,sum = " << sum << std::endl;
	return 0;
}
