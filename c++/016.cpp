#include <iostream>
#include <thread>
#include <mutex>
using namespace std;
int main()
{
	int sum = 0;
	mutex mtx;
	cout << "Before joining,sum = " << sum << std::endl;
	thread t1([&]{
		for (int i = 0; i < 100000; ++i)
		{
			//通过加锁来保证原子性
			mtx.lock();
			++sum;
			mtx.unlock();
		}
	});
	thread t2([&]{
		for (int i = 0; i < 100000; ++i)
		{
			mtx.lock();
			++sum;
			mtx.unlock();
		}
	});

	t1.join();
	t2.join();
	cout << "After joining,sum = " << sum << std::endl;
	return 0;
}
