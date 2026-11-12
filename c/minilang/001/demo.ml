// 第 1 步：看看词法分析器能认出哪些 token
let x = 10;
let y = 3.14;
let name = "MiniLang";
// 注释会被跳过
fn add(a, b) { return a + b; }
if (x > 1) { print("big\n"); } else { print("small\n"); }
for (let i = 0; i < 3; i = i + 1) { print("{i}\n"); }
