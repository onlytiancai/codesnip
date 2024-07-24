
#include <memory>
class Node {
public:
    std::shared_ptr<Node> next;
    std::weak_ptr<Node> prev;
    // ...其他成员和方法
};

void createChain() {
    auto node1 = std::make_shared<Node>();
    auto node2 = std::make_shared<Node>();

    node1->next = node2;
    node2->prev = node1; // 使用weak_ptr避免循环引用
}

int main() {
    createChain();
    // 所有资源在离开作用域时将被正确释放，无内存泄漏风险
    return 0;
}
