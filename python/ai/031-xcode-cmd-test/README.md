# 031-xcode-cmd-test

检测本机是否安装了 **Xcode Command Line Tools (CLT)**，并通过 Swift + AppKit GUI Hello World 做端到端验证。**GUI 以 `.app` bundle 形式启动**，带自定义 Dock 图标、ad-hoc 代码签名和 Bundle ID。

GUI 能弹窗 ⇒ CLT（含 macOS SDK + AppKit 框架）可用 ⇒ 检测脚本结论正确。

## 文件

| 文件 | 作用 |
|---|---|
| `detect.sh` | 单独运行，检测 CLT 关键组件 |
| `HelloWorld.swift` | Swift + AppKit GUI 源 |
| `Info.plist` | bundle 元数据（Bundle ID、最低系统版本、Dock 图标声明） |
| `Resources/AppIcon.icns` | Dock 图标（来自系统 Finder.icns） |
| `build.sh` | 检测 + 编译 + 打包 .app + 签名 + 启动 GUI |
| `HelloWorld.app/` | build 产物（gitignore） |
| `docs/macos-app-bundle.md` | bundle vs 单文件二进制理论 |
| `docs/macos-sdk-capabilities.md` | macOS SDK 能力速览（按"做什么"分块） |
| `docs/bundle-vs-single.md` | bundle 化前后的实测对比 |

## 用法

```bash
chmod +x detect.sh build.sh
./build.sh
```

启动后：
- 终端打印 4 个阶段的进度（检测 / 编译 / 打包签名 / 启动）
- 弹窗 + Dock 出现 Finder 图标（因为 `AppIcon.icns` 是 Finder 图标）
- 窗口里显示 Bundle ID `com.example.HelloWorld` 和 Resources 路径

### 仅检测（不编译不弹窗）

```bash
./detect.sh
```

退出码：`0` = 全部可用，`1` = 缺失或损坏。

### 只看 .app 结构（已编译过的话）

```bash
./build.sh   # 已 build 过会重新打包
find HelloWorld.app -type f | sort
```

### 验证 Bundle ID 和签名

```bash
/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" HelloWorld.app/Contents/Info.plist
codesign -dv HelloWorld.app
```

## Bundle 结构

```
HelloWorld.app/
└── Contents/
    ├── Info.plist                   # bundle 元数据
    ├── PkgInfo                      # 8 字节老式描述（APPL????）
    ├── MacOS/
    │   └── HelloWorld               # 可执行
    └── Resources/
        └── AppIcon.icns             # Dock 图标
```

## 预期输出

### detect.sh

```
=== Xcode Command Line Tools 检测 ===
✓ xcode-select 路径:    /Library/Developer/CommandLineTools
✓ swiftc:                /usr/bin/swiftc
✓ xcrun swiftc:          /Library/Developer/CommandLineTools/usr/bin/swiftc
✓ macOS SDK 路径:        /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
ℹ macOS SDK 版本:        27.0
ℹ Swift:                 Apple Swift version 6.x
=== 全部组件可用 ===
```

### build.sh

```
=== 阶段 2/4: 编译 ===
✓ 编译成功: build/HelloWorld

=== 阶段 3/4: 打包 + 签名 ===
✓ Ad-hoc 签名成功
✓ Bundle 结构:
    HelloWorld.app/Contents/Info.plist
    HelloWorld.app/Contents/MacOS/HelloWorld
    HelloWorld.app/Contents/PkgInfo
    HelloWorld.app/Contents/Resources/AppIcon.icns

=== 阶段 4/4: 启动 GUI ===
GUI 进程 PID: xxxxx
```

### GUI

弹出 520×320 窗口：
- **Hello, World!**（28pt 粗体）
- Xcode CLT 检测通过 ✓（绿色）
- **Bundle ID: com.example.HelloWorld**（蓝色，bundle 化特有）
- macOS SDK 版本
- CLT 路径
- Resources 路径
- 「退出」按钮

Dock 里出现 Finder 图标。

## 检测覆盖

| 检查项 | 失败影响 |
|---|---|
| `xcode-select -p` 返回有效路径 | exit 1 |
| `swiftc` 在 PATH | exit 1 |
| `xcrun --find swiftc` 能定位 | exit 1 |
| `xcrun --show-sdk-path --sdk macosx` | exit 1 |
| `xcrun --show-sdk-version` | 仅警告 |
| `swift --version` | 仅警告 |

## 常见问题

### 「xcode-select 无法定位 CLT」

```bash
xcode-select --install
```

### 「codesign 失败」

```bash
which codesign    # 应返回 /usr/bin/codesign
# 如果不在，重新装 CLT
```

### 「`open` 启动后窗口不出来」

可能 Info.plist 格式错：

```bash
plutil -lint Info.plist
```

### Dock 图标不显示

确认没设 `LSUIElement=true`（那个字段会让 App 变成菜单栏应用，Dock 不显示）。本项目 Info.plist 没设这个字段。

### 资源加载返回 nil

确认 `AppIcon.icns` 在 `HelloWorld.app/Contents/Resources/` 下：

```bash
ls HelloWorld.app/Contents/Resources/
```

## 本机环境（参考）

- macOS 27.0
- CLT 路径：`/Library/Developer/CommandLineTools`
- Apple Silicon (M4)

## 学习要点

1. **Bundle 本质**：一个目录 + Info.plist + 可执行 + 可选资源
2. **Bundle ID**：进程身份、UserDefaults、IPC、entitlements 全靠它
3. **资源加载**：`Bundle.main.url(forResource:withExtension:)` 是入口
4. **代码签名**：ad-hoc (`--sign -`) 本机够用；分发需要开发者证书 + 公证
5. **Info.plist 字段**：CFBundle* 一族 + LSMinimumSystemVersion + NS* AppKit 专属

详见 `docs/bundle-vs-single.md` 实测对比。
