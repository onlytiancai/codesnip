# macOS `.app` Bundle vs 单文件二进制

本项目（`031-xcode-cmd-test/`）的产物 `./HelloWorld` 是一个 Mach-O 单文件二进制，**不是** `.app` bundle。本文档解释两者的区别，以及什么时候必须升级到 bundle。

## 一句话区别

| | 单文件二进制 | `.app` Bundle |
|---|---|---|
| 本质 | 一个可执行文件 | 一个**伪装成文件的目录**（Finder 里看着像单文件） |
| macOS 对待 | CLI 进程 | GUI 应用 |
| 启动方式 | `./HelloWorld` | `open HelloWorld.app` / Finder 双击 |
| 身份标识 | 文件名 | `Info.plist` 里的 `CFBundleIdentifier` |
| Dock 图标 | 临时，需要手动调 `setActivationPolicy(.regular)` | 自动、永久，可自定义 |
| 代码签名 / Gatekeeper | 没有 | 需要（分发或上架必备） |
| 沙盒 / entitlements | 不支持 | 支持 |

## 单文件二进制 `./HelloWorld`

```
HelloWorld               # 1 个 Mach-O 可执行
```

- macOS 把它当**普通 CLI 进程**对待（和 `ls`、`git` 同类）
- 进程名 = 文件名
- 没有"Dock 图标自定义"，没有版本号，没有 bundle ID
- 很多 AppKit 周边 API 会退化成"以可执行文件名猜身份"：
  - `UserDefaults.standard` 用 `HelloWorld` 当 suite name
  - `FileManager.urls(for: .applicationSupportDirectory)` 返回 `~/Library/Application Support/HelloWorld/`
  - 不能注册 URL scheme、文件类型关联
- 不参与 Gatekeeper 校验、App Store、公证（Notarization）

## `.app` Bundle

```
HelloWorld.app/                       # ← 整个目录被 Finder 当成"应用"
└── Contents/
    ├── Info.plist                    # 必需：bundle 元数据
    ├── PkgInfo                       # 可选：8 字节老式描述
    ├── MacOS/
    │   └── HelloWorld                # 真正的可执行
    ├── Resources/                    # 可选：.icns 图标、nib、字体、本地化
    ├── Frameworks/                   # 可选：内嵌私有 dylib
    └── _CodeSignature/               # 可选：代码签名
```

### `Info.plist` 关键字段

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleIdentifier</key>             <string>com.example.HelloWorld</string>
    <key>CFBundleName</key>                   <string>HelloWorld</string>
    <key>CFBundleExecutable</key>             <string>HelloWorld</string>
    <key>CFBundlePackageType</key>            <string>APPL</string>
    <key>CFBundleShortVersionString</key>     <string>1.0</string>
    <key>CFBundleVersion</key>                <string>1</string>
    <key>LSMinimumSystemVersion</key>         <string>13.0</string>
    <key>NSHighResolutionCapable</key>       <true/>
</dict>
</plist>
```

## 什么时候单文件够用

- CLI 工具、后台 daemon、测试 demo
- **本项目**（验证 Xcode CLT 是否可用）：编译能跑 + 窗口能弹就够
- 内部脚本，开发自己用

## 什么时候必须 `.app`

- 拖到 `/Applications` 给别人用
- 上 App Store
- 需要 Gatekeeper 通过（公证 + 签名）
- 需要沙盒、entitlements（摄像头/麦克风/文件访问/网络受限）
- 需要稳定的 bundle ID（用于 UserDefaults、Application Support、IPC、URL scheme 注册）
- 想自定义 Dock 图标和 About 面板
- 想内嵌私有 framework（避免依赖系统版本）

## 从单文件升级到 `.app`

把现有 `HelloWorld` 二进制打包成 bundle 的最小步骤：

```bash
APP=HelloWorld.app
mkdir -p "$APP/Contents/MacOS"
cp HelloWorld "$APP/Contents/MacOS/"

# 写 Info.plist（见上面模板）

# ad-hoc 签名（可选，但本机跑更稳）
codesign --force --deep --sign - "$APP"

# 启动
open "$APP"
```

`.app` 化的 `swiftc` 编译命令本身**不变**——bundle 只是事后包装。也可以在 `build.sh` 里加 `bundle` 步骤：

```bash
APP=HelloWorld.app
mkdir -p "$APP/Contents/MacOS"
cp HelloWorld "$APP/Contents/MacOS/"
cat > "$APP/Contents/Info.plist" <<'EOF'
...（plist 内容）
EOF
codesign --force --deep --sign - "$APP"
```

## 参考

- Apple: [Bundle Programming Guide](https://developer.apple.com/library/archive/documentation/CoreFoundation/Conceptual/CFBundles/BundleTypes/BundleTypes.html)（归档但仍是权威）
- `man codesign` / `man open`
- `plutil -lint Info.plist`：检查 plist 语法
