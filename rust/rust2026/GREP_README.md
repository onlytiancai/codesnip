# Rust Grep 工具

这是一个用 Rust 实现的简单 grep 工具，支持正则表达式搜索和多文件处理。

## 功能特性

- 支持正则表达式模式搜索
- 多文件搜索
- 显示行号
- 忽略大小写搜索
- 反转匹配（显示不匹配的行）
- 统计匹配行数
- 只显示匹配部分

## 安装

```bash
cargo build --release
```

## 使用方法

### 基本用法

```bash
# 在文件中搜索模式
cargo run -- <pattern> <file>

# 在多个文件中搜索
cargo run -- <pattern> <file1> <file2>

# 从标准输入读取
echo "some text" | cargo run -- <pattern>
```

### 命令行选项

- `-l, --line-number`: 显示行号
- `-i, --ignore-case`: 忽略大小写
- `-v, --invert-match`: 反转匹配（显示不匹配的行）
- `-c, --count`: 只显示匹配行数
- `-o, --only-matching`: 只显示匹配的部分

### 使用示例

1. **基本搜索**
```bash
cargo run -- "Hello" test_file.txt
```

2. **显示行号**
```bash
cargo run -- -l "Rust" test_file.txt
```

3. **忽略大小写**
```bash
cargo run -- -i "hello" test_file.txt
```

4. **反转匹配**
```bash
cargo run -- -v "test" test_file.txt
```

5. **统计匹配行数**
```bash
cargo run -- -c "test" test_file.txt
```

6. **只显示匹配部分**
```bash
cargo run -- -o "test" test_file.txt
```

7. **使用正则表达式**
```bash
cargo run -- "\d+" test_file.txt
```

8. **组合选项**
```bash
cargo run -- -i -l "hello" test_file.txt
```

## 实现细节

- 使用 `clap` 库进行命令行参数解析
- 使用 `regex` 库进行正则表达式匹配
- 支持从标准输入读取或从文件读取
- 优雅的错误处理和友好的错误信息
- 多文件搜索时显示文件名前缀

## 项目结构

```
src/
├── main.rs     # 程序入口
└── grep.rs     # grep 工具实现
```

## 依赖

- `clap`: 命令行参数解析
- `regex`: 正则表达式支持
- `itertools`: 迭代器工具（已在项目中）
