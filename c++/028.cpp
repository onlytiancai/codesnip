#include <memory>

void manageResource(std::unique_ptr<int> ptr) {
    // 使用资源
} // ptr在此处自动销毁，资源被释放

int main() {
    auto ptr = std::make_unique<int>(42); // 创建并初始化unique_ptr
    manageResource(std::move(ptr)); // 移动所有权到函数内
    // ptr现在为空，资源已在manageResource内部被释放
    return 0;
}
