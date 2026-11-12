# Bundle vs 单文件：实测对比

本文用 `031-xcode-cmd-test/` 项目的实际行为，对比 `.app` bundle 和单文件 Mach-O 在 macOS 下的差异。读完你应该能回答：**我的项目到底要不要打成 bundle？**

## 对比表

| 维度 | 单文件二进制 | `.app` bundle |
|---|---|---|
| 本质 | 一个 Mach-O 文件 | 一个**伪装成文件的目录** |
| macOS 看待 | CLI 进程 | 正经 GUI App |
| 启动方式 | `./HelloWorld` | `open HelloWorld.app` 或 Finder 双击 |
| 进程名 | `HelloWorld` | `HelloWorld`（同名） |
| `Bundle.main.bundleIdentifier` | `nil` | `com.example.HelloWorld` |
| `Bundle.main.resourcePath` | `nil` | `/.../HelloWorld.app/Contents/Resources` |
| Dock 图标 | 默认 generic | `AppIcon.icns`（资源加载成功才有） |
| UserDefaults plist 路径 | `~/Library/Preferences/HelloWorld.plist` | `~/Library/Preferences/com.example.HelloWorld.plist` |
| `NSApp.activate` 行为 | 需要显式调 | 自动前台 |
| `NSWorkspace.runningApplications` 查找 | 找不到（无 bundle ID） | 按 bundle ID 找到 |
| codesign | 可选 | 默认必备（launchd 区分正常 App vs 灰色 App） |
| Apple Events / AppleScript 控制 | 受限 | 完整 |

## 怎么验证

### 1. Bundle ID

```bash
# Bundle 内
/usr/libexec/PlistBuddy -c "Print :CFBundleIdentifier" HelloWorld.app/Contents/Info.plist
# → com.example.HelloWorld

# 运行时
./build.sh &
sleep 3
ps -ax | grep HelloWorld
# 看到进程名是 HelloWorld，但 Bundle.main.bundleIdentifier 在 GUI 窗口里显示
```

### 2. UserDefaults 路径对比

```bash
# 单文件版本（如果回退到 ./HelloWorld 运行）
defaults read HelloWorld 2>/dev/null
# → plist: ~/Library/Preferences/HelloWorld.plist

# Bundle 版本（当前）
defaults read com.example.HelloWorld 2>/dev/null
# → plist: ~/Library/Preferences/com.example.HelloWorld.plist
```

路径里的"HelloWorld"和"com.example.HelloWorld"就是 bundle ID 起的作用。

### 3. 资源加载

单文件运行：

```bash
swiftc -framework Cocoa -o /tmp/hw HelloWorld.swift
/tmp/hw &
# 窗口里 Bundle ID 那行会显示 "(no bundle ID — CLI 进程)"
# Resources 那行会显示 "(no resourcePath)"
# Dock 图标是 generic
```

Bundle 运行：

```bash
./build.sh
# 窗口里 Bundle ID 显示 com.example.HelloWorld
# Resources 显示 .../HelloWorld.app/Contents/Resources
# Dock 图标是 Finder 图标
```

### 4. 代码签名

```bash
# 签名状态
codesign -dv HelloWorld.app
# → Identifier=com.example.HelloWorld
# → Format=app bundle with Mach-O universal [...]
# → Signature=adhoc

# 详细
codesign -dvv HelloWorld.app
```

### 5. Launch Services 注册

Bundle 启动后，macOS 的 Launch Services 会把它注册到 `~/Library/Preferences/com.apple.LaunchServices/com.apple.launchservices.secure.plist`，所以：

```bash
lsregister -dump | grep -A1 -i helloworld | head -20
# 看到 com.example.HelloWorld 出现在 bundle 列表里
```

这就是为什么双击 `.app` Finder 知道用 launchd 启动它，而不是直接 exec 二进制。

## 真实使用场景选择

### 选单文件

- CLI 工具（`mygit`、`aws` 这种）
- 后台 daemon
- 临时测试 / 学习
- 不想搞 bundle 结构，专注于 AppKit 代码本身

### 选 bundle

- 想自定义 Dock 图标
- 需要稳定 bundle ID（UserDefaults / IPC / 文件关联）
- 想被 AppleScript 控制
- 想上架或分发给其他机器
- 想用系统通知、辅助功能、屏幕录制等需要 entitlements 的功能
- 想被 Finder 当 App 看待

## Bundle 化的代价

为了一个 `com.example.HelloWorld` 这个身份，你付出：

- 至少 5 个文件（Info.plist + PkgInfo + 可执行 + 至少 0 个资源）
- 至少 1 个目录层级（`Contents/{MacOS,Resources}`）
- 必须用 `open` 或 Finder 启动（直接 exec 也能跑但 launchd 行为退化）
- 必须 `codesign`（哪怕 ad-hoc）

回报：进程身份、Dock 图标、UserDefaults 路径、AppleScript 控制权，**全部**。

## 关键决策点速查

```
问自己：我的程序需要 bundle ID 吗？
├─ 否 → 单文件
└─ 是 → 继续问
   ├─ 需要自定义图标？→ bundle（含 Resources/）
   ├─ 需要被 AppleScript 控制？→ bundle + Info.plist 完整字段
   ├─ 需要分发给其他人？→ bundle + 签名 + 公证
   └─ 只是 NSWorkspace 找得到？→ bundle（最小化也够）
```

## 参考

- 本项目 `docs/macos-app-bundle.md`：bundle 结构与字段详解
- `man codesign` / `man open`
- `plutil -lint Info.plist`：plist 语法检查
- `lsregister -dump`：查看 Launch Services 注册表
