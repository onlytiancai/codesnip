#include <iostream>           // std::cout
#include <thread>             // std::thread
#include <mutex>              // std::mutex, std::unique_lock
#include <condition_variable> // std::condition_variable

std::mutex mtx;
std::condition_variable cv;
bool ready = false;

void print_id(int id) {
    std::unique_lock<std::mutex> lck(mtx);
    while (!ready) cv.wait(lck);
    // ...
    std::cout << "thread " << id << '\n';
}

void go() {
    std::unique_lock<std::mutex> lck(mtx);
    ready = true;
    cv.notify_all();
}

int main()
{
    std::thread threads[10];
    // spawn 10 threads:
    for (int i = 0; i<10; ++i)
        threads[i] = std::thread(print_id, i);
        
    std::cout << "10 threads ready to race...\n";
    go();                       // go!
    
    for (auto& th : threads) th.join();

    return 0;
}

//以上的代码，在12行中调用cv.wait(lck)的时候，线程将进入休眠。
//在调用31行的go函数之前，10个线程都处于休眠状态。
//当20行的cv.notify_all()运行后，12行的休眠将结束，继续往下运行，最终输出如上结果。

