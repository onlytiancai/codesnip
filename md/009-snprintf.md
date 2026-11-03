# C 语言函数 `snprintf()` 详解

`snprintf()` 是 C 语言标准库中的一个格式化输出函数，定义在头文件 `<stdio.h>` 中。

它的主要作用是：将格式化后的字符串写入指定的字符数组中，并限制最多写入的字符数量，避免缓冲区溢出。

相比 `sprintf()`，`snprintf()` 更安全，因为它可以控制写入的最大长度。

## 1. 函数原型

```
#include <stdio.h>

int snprintf(char *str, size_t size,
             const char *format, ...);
```

参数说明：

| 参数       | 含义            |
| -------- | ------------- |
| `str`    | 目标字符串缓冲区      |
| `size`   | 缓冲区的总大小，单位是字节 |
| `format` | 格式化字符串        |
| `...`    | 需要格式化的变量      |

返回值：

- 返回值 ≥ 0：表示本来需要输出的字符数量，不包含结尾的 `'\0'`。
- 返回值 ≥ `size`：表示输出被截断了。
- 返回值 < 0：表示发生了编码或输出错误。

注意：`size` 包含字符串结尾的 `'\0'` 所占用的空间。

## 2. 基本使用示例

### 示例一：格式化整数和字符串

```
#include <stdio.h>

int main(void)
{
    char buf[100];
    int age = 25;
    char name[] = "Tom";

    snprintf(buf, sizeof(buf),
             "Name: %s, Age: %d",
             name, age);

    printf("%s\n", buf);

    return 0;
}
```

输出：

```
Name: Tom, Age: 25
```

执行过程：

1. `snprintf()` 将 `name` 和 `age` 按照格式字符串进行转换。
2. 将转换后的内容写入 `buf`。
3. 自动在末尾添加 `'\0'`。
4. `printf()` 将 `buf` 的内容打印出来。

### 示例二：限制输出长度

```
#include <stdio.h>

int main(void)
{
    char buf[10];

    int ret = snprintf(buf, sizeof(buf),
                       "Hello World");

    printf("buf = %s\n", buf);
    printf("ret = %d\n", ret);

    return 0;
}
```

输出：

```
buf = Hello Wo
ret = 11
```

为什么？

缓冲区大小为 10 字节：

| 内容         | 占用空间 |
| ---------- | ---- |
| `Hello Wo` | 8 字节 |
| `r`        | 1 字节 |
| `l`        | 1 字节 |
| `d`        | 1 字节 |
| `'\0'`     | 1 字节 |

原始字符串 `"Hello World"` 长度是 11。

但是 `buf` 只有 10 字节，所以只能存储 9 个字符，再加上字符串结束符 `'\0'`。

最终：

```
buf[0] ~ buf[7] = Hello Wo
buf[8]           = '\0'
```

返回值仍然是 11，因为它表示完整格式化结果所需要的字符数，而不是实际写入的字符数。

[image](https://www.google.com/s2/favicons?domain=https://en.cppreference.com\&sz=32)

cppreference.com+1



## 3. 如何判断是否发生截断？

这是 `snprintf()` 最重要的使用技巧。

```
#include <stdio.h>

int main(void)
{
    char buf[10];

    int ret = snprintf(buf, sizeof(buf),
                       "Hello World");

    if (ret < 0)
    {
        printf("Formatting error\n");
    }
    else if ((size_t)ret >= sizeof(buf))
    {
        printf("Output truncated\n");
    }
    else
    {
        printf("Output complete\n");
    }

    return 0;
}
```

判断逻辑：

```
ret < 0
```

表示格式化失败。

```
ret >= sizeof(buf)
```

表示输出被截断。

```
ret < sizeof(buf)
```

表示输出完整。

注意：这里使用 `size_t` 类型转换，是为了避免有符号和无符号整数比较产生警告。

## 4. 常用格式化符号

| 格式符   | 含义          | 示例         |
| ----- | ----------- | ---------- |
| `%d`  | 十进制整数       | `123`      |
| `%u`  | 无符号整数       | `123`      |
| `%x`  | 十六进制小写      | `ff`       |
| `%X`  | 十六进制大写      | `FF`       |
| `%f`  | 浮点数         | `3.140000` |
| `%s`  | 字符串         | `hello`    |
| `%c`  | 单个字符        | `A`        |
| `%p`  | 指针地址        | `0x1234`   |
| `%zu` | `size_t` 类型 | `100`      |
| `%%`  | 输出百分号       | `%`        |

例如：

```
char buf[100];

int num = 255;

snprintf(buf, sizeof(buf),
         "num=%d, hex=0x%X",
         num, num);
```

结果：

```
num=255, hex=0xFF
```

## 5. `snprintf()` 和 `sprintf()` 的区别

| 比较            | `sprintf()` | `snprintf()`  |
| ------------- | ----------- | ------------- |
| 是否限制缓冲区大小     | 否           | 是             |
| 是否可能缓冲区溢出     | 是           | 正确使用时不会       |
| 是否自动添加 `'\0'` | 是           | 是（size > 0 时） |
| 是否返回完整字符串长度   | 是           | 是             |
| 推荐使用          | 不推荐         | 推荐            |

例如：

```
char buf[10];

sprintf(buf, "%s", "Hello World");
```

这个例子会导致缓冲区溢出，属于未定义行为。

而：

```
snprintf(buf, sizeof(buf), "%s", "Hello World");
```

则会限制写入长度。

## 6. 实际项目中的应用

### 拼接字符串

```
char path[128];

const char *dir = "/home/user";
const char *file = "test.txt";

snprintf(path, sizeof(path),
         "%s/%s", dir, file);
```

结果：

```
/home/user/test.txt
```

### 格式化十六进制数据

在嵌入式开发、串口通信、协议调试中，经常需要把字节转换成十六进制字符串。

```
#include <stdio.h>

int main(void)
{
    unsigned char data = 0xAB;
    char buf[8];

    snprintf(buf, sizeof(buf),
             "0x%02X", data);

    printf("%s\n", buf);

    return 0;
}
```

输出：

```
0xAB
```

这里：

- `%X`：输出十六进制大写。
- `%02X`：至少输出两位，不足两位用 `0` 补齐。

例如：

```
snprintf(buf, sizeof(buf), "%02X", 5);
```

输出：

```
05
```

## 7. 进阶：动态计算需要的缓冲区大小

如果不知道最终字符串有多长，可以先计算长度。

```
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    int num = 12345;

    int len = snprintf(NULL, 0,
                       "Number: %d", num);

    if (len < 0)
        return 1;

    char *buf = malloc((size_t)len + 1);

    if (buf == NULL)
        return 1;

    snprintf(buf, (size_t)len + 1,
             "Number: %d", num);

    printf("%s\n", buf);

    free(buf);

    return 0;
}
```

核心原理：

```
int len = snprintf(NULL, 0, format, ...);
```

此时不写入任何数据，只计算完整输出需要多少字符。

然后：

```
malloc(len + 1);
```

多分配一个字节，用于存放字符串结束符 `'\0'`。

这种方式适合生成长度不确定的字符串。

[image](https://www.google.com/s2/favicons?domain=https://en.cppreference.com\&sz=32)

cppreference.com+1



## 8. 使用时需要特别注意

① `size` 是字节数，不是字符数。

例如：

```
char buf[10];

snprintf(buf, sizeof(buf), "你好世界");
```

如果使用 UTF-8 编码，中文通常每个汉字占 3 个字节，因此这里可能只能写入部分中文字符。

② 不要让源字符串和目标缓冲区重叠。

错误示例：

```
char buf[100] = "Hello";

snprintf(buf, sizeof(buf),
         "%s World", buf);
```

这种源目标重叠的行为是未定义的。

[image](https://www.google.com/s2/favicons?domain=https://en.cppreference.com\&sz=32)

cppreference.com+1



③ `snprintf()` 不是万能的安全函数。

格式化参数的类型必须匹配。例如：

```
int num = 100;

snprintf(buf, sizeof(buf), "%f", num);
```

这是错误的用法，因为 `%f` 需要 `double` 类型，而不是 `int`。

总结：

`snprintf()` 最核心的三个知识点：

1. `size` 控制目标缓冲区的最大容量。
2. 返回值表示完整输出所需的字符数，不包含 `'\0'`。
3. 通过 `ret >= size` 判断是否发生了截断。

可以记住这个常用模板：

```
char buf[128];

int ret = snprintf(buf, sizeof(buf),
                   "value=%d", 123);

if (ret < 0 || (size_t)ret >= sizeof(buf))
{
    // 错误处理或输出截断处理
}
```

参考：[cppreference：snprintf](https://en.cppreference.com/w/c/io/printf?utm_source=chatgpt.com)