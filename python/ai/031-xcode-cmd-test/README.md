# 031-xcode-cmd-test

检测本机是否安装了 **Xcode Command Line Tools (CLT)**，并通过一个 Swift + AppKit GUI Hello World 做端到端验证。

GUI 能弹窗 ⇒ CLT（含 macOS SDK + AppKit 框架）可用 ⇒ 检测脚本的结论正确。

## 文件

| 文件 | 作用 |
|---|---|
| `detect.sh` | 单独运行，检测 CLT 关键组件 |
| `HelloWorld.swift` | Swift + AppKit GUI 源 |
| `build.sh` | 检测 + 编译 + 启动 GUI 一键脚本 |

## 用法

```bash
chmod +x detect.sh build.sh
./build.sh
```

### 仅检测（不编译不弹窗）

```bash
./detect.sh
```

退出码：`0` = 全部可用，`1` = 缺失或损坏。

### 单独编译运行

```bash
swiftc -target arm64-apple-macosx13.0 -o HelloWorld -framework Cocoa HelloWorld.swift
./HelloWorld
```

> Intel Mac 改 `x86_64-apple-macosx13.0`。

## 预期输出

### detect.sh

```
=== Xcode Command Line Tools 检测 ===
✓ xcode-select 路径:    /Library/Developer/CommandLineTools
✓ swiftc:                /usr/bin/swiftc
✓ xcrun swiftc:          /Library/Developer/CommandLineTools/usr/bin/swiftc
✓ macOS SDK 路径:        /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
ℹ macOS SDK 版本:        26.0
ℹ Swift:                 Apple Swift version 6.x
=== 全部组件可用 ===
```

### GUI

弹出一个 460×280 的窗口，包含：
- **Hello, World!**（28pt 粗体，居中）
- Xcode CLT 检测通过 ✓（绿色）
- macOS SDK 版本
- CLT 路径
- 「退出」按钮

## 检测覆盖

| 检查项 | 失败影响 |
|---|---|
| `xcode-select -p` 返回有效路径 | exit 1 |
| `swiftc` 在 PATH | exit 1 |
| `xcrun --find swiftc` 能定位 | exit 1（区分 PATH 命中但 xcrun 拒绝）|
| `xcrun --show-sdk-path --sdk macosx` | exit 1 |
| `xcrun --show-sdk-version` | 仅警告 |
| `swift --version` | 仅警告 |

## 常见问题

### 「xcode-select 无法定位 CLT」

```bash
xcode-select --install
```

### 「xcrun 找不到 swiftc」

CLT 损坏，重装：

```bash
sudo rm -rf /Library/Developer/CommandLineTools
xcode-select --install
```

### 编译报 `error: unable to find SDK`

SDK 路径异常，确认 `xcrun --show-sdk-path --sdk macosx` 能返回真实目录。

## 本机环境（参考）

- macOS 27.0
- CLT 路径：`/Library/Developer/CommandLineTools`
- Apple Silicon (M4)
