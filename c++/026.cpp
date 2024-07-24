#include <iostream>
#include <list>

int main()
{

	std::list< std::pair<int, std::string> > mylist;

	std::pair<int, std::string> kv(1, "11111");
	mylist.push_back(kv);
	mylist.emplace_back(kv);
    std::cout << std::endl;

	mylist.push_back(std::make_pair(2, "sort"));
	mylist.push_back({ 40, "sort" });

    std::cout << std::endl;
	mylist.emplace_back(std::make_pair(2, "sort"));
	mylist.emplace_back(10, "sort");

    for (auto e : mylist)
		std::cout << e.first << ":" << e.second << std::endl;

	return 0;
}
