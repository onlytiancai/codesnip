#include<iostream>
#include<mutex>
#include<condition_variable>
#include<thread>
using namespace std;


//用互斥锁和条件变量实现交替打印数据
int main()
{
	int n = 100;
	int i = 0;
	condition_variable cv;
	mutex mtx;
	bool flag = false;

	//打印奇数
	thread t1([&]{
		while (i < n)
		{
			//加锁
			unique_lock<mutex> lock(mtx);

			//flag是false时会一直处于阻塞状态，直到flag变为true
			cv.wait(lock, [&]{return !flag; });

			cout << std::this_thread::get_id() << "->" << i << endl;
			++i;

			//防止该线程在自己的时间片内多次成功竞争锁，多次打印
			flag = true;
			//唤醒其他线程(一个)
			cv.notify_one();
		}
	});

	//打印偶数
	thread t2([&]{
		while (i < n)
		{
			unique_lock<mutex> lock(mtx);
			cv.wait(lock, [&]{return flag; });
			cout << std::this_thread::get_id() << "->" << i << endl;
			++i;
			flag = false;
			cv.notify_one();
		}
	});

	t1.join();
	t2.join();

	return 0;
}
