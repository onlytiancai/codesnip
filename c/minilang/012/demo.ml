// 第 12 步：一个正常跑通的程序
// （想看错误长什么样？试试把最后一行改成 print(nosuch); ）
fn max2(a, b) {
    if (a > b) {
        return a;
    }
    return b;
}
let x = max2(3, 9);
print("max = {x}\n");
