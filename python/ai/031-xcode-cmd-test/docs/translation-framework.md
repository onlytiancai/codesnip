# Translation framework 学习笔记

> 适用：macOS 15.0+ / iOS 18.0+。本机 SDK 27.0，本项目 target 锁 15.0。

## 它是什么

Apple 在 macOS 15 / iOS 18 引入的**系统级本地化翻译 framework**。不同于 Google Translate / DeepL 那种云端 API，它是**离线**的——首次用某语种对时下载本地翻译模型，之后无需网络。

## 能做什么

- 把任意文本翻译到目标语种
- 支持 100+ 语种
- 自动识别语种对支持状态（已下载 / 支持但未下载 / 不支持）
- `AttributedString` 翻译（macOS 26.4+），保留样式
- 批量翻译（async sequence）
- 与系统右键菜单的「翻译」共享同一个翻译引擎

## 核心 API（macOS 15.0+ base）

```swift
import Translation

// 1. 检查语种对支持（不依赖 session）
let availability = LanguageAvailability()
let status = await availability.status(
    from: Locale.Language(identifier: "en"),
    to: Locale.Language(identifier: "zh-Hans")
)
// status: .installed / .supported / .unsupported

// 2. 构造 Configuration（macOS 15+）
let configuration = TranslationSession.Configuration(
    source: Locale.Language(identifier: "en"),
    target: Locale.Language(identifier: "zh-Hans")
)

// 3. 拿 session——只能在 SwiftUI `.translationTask` 里
struct MyView: View {
    var body: some View {
        Text("hello")
            .translationTask(configuration) { session in
                let response = try await session.translate("Hello, World!")
                print(response.targetText)  // "你好，世界！"
            }
    }
}
```

### 重要：API 路径分两条

| API | 所属类型 | 适用场景 | Deployment |
|---|---|---|---|
| `Configuration(source:target:)` | `TranslationSession.Configuration` | 任何 macOS 15+ | macOS 15.0+ |
| `.translationTask(configuration)` | SwiftUI `View` | SwiftUI 项目 | macOS 15.0+ |
| `TranslationSession(installedSource:target:)` | `TranslationSession` | 非 SwiftUI 直接构造 | macOS 26.0+ |

**坑**：`TranslationSession.init(source:target:)` 这个 API **不存在**——`init(source:target:)` 属于 `Configuration`，不是 session 直接 init。我之前就是这么被骗的，以为可以直接 `TranslationSession(source:...)`。

macOS 15.0 上**唯一**拿到 session 的方式是 `.translationTask(configuration)`，它是 SwiftUI `View` 的 modifier。

## AppKit 项目怎么拿 session：NSHostingView 桥接

因为 `.translationTask` 是 SwiftUI 专属，AppKit 项目必须桥接。流程：

1. 写一个 SwiftUI view 用 `.translationTask(configuration)` 拿 session
2. 用 `NSHostingView(rootView: ...)` 把这个 SwiftUI view 嵌进 AppKit 窗口
3. 桥接对象（普通引用类型）在两边共享，AppKit 写入，SwiftUI 闭包读

### 桥接代码模式

```swift
// 1. 桥接对象（普通类，不需要 @Observable）
@available(macOS 15.0, *)
final class TranslationBridge {
    let configuration: TranslationSession.Configuration
    let text: String
    var lastResponse: TranslationSession.Response?
    var lastError: String?

    init(configuration: TranslationSession.Configuration, text: String) {
        self.configuration = configuration
        self.text = text
    }
}

// 2. SwiftUI 容器（只负责拿 session，不显示）
@available(macOS 15.0, *)
struct TranslationBridgeView: View {
    let bridge: TranslationBridge

    var body: some View {
        EmptyView()
            .translationTask(bridge.configuration) { session in
                do {
                    let response = try await session.translate(bridge.text)
                    bridge.lastResponse = response
                } catch {
                    bridge.lastError = String(describing: error)
                }
            }
    }
}

// 3. AppKit 端：嵌 hosting view + 轮询结果
@available(macOS 15.0, *)
final class TranslationView: NSView {
    private var bridge: TranslationBridge?
    private var bridgeHostingView: NSHostingView<TranslationBridgeView>?
    private var pollTimer: Timer?

    func doTranslate(source: Locale.Language, target: Locale.Language, text: String) {
        // 每次新建 bridge + 替换 hosting view 的 rootView，
        // 让 SwiftUI 重建 view 树并重新执行 .translationTask
        let bridge = TranslationBridge(
            configuration: TranslationSession.Configuration(source: source, target: target),
            text: text
        )
        self.bridge = bridge
        if let hosting = bridgeHostingView {
            hosting.rootView = TranslationBridgeView(bridge: bridge)
        } else {
            let hosting = NSHostingView(rootView: TranslationBridgeView(bridge: bridge))
            hosting.frame = .zero
            addSubview(hosting)
            bridgeHostingView = hosting
        }
        startPolling()
    }

    private func startPolling() {
        guard let bridge = self.bridge else { return }
        pollTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            // 检查 bridge.lastResponse / lastError，写到 AppKit UI
        }
    }
}
```

### 为什么不直接用 @Observable / @State

`@Observable` / `@State` 都是 SwiftUI 编译器宏，纯 CLT（Command Line Tools）的 swiftc **不能加载** SwiftUI 的 macros plugin——必须有 Xcode。**这是本项目踩过的坑**。

绕过方案：用 **替换 rootView** 代替 @State 触发 SwiftUI 重建；用**普通 class** 代替 @Observable 当桥接对象。每次新建 bridge 对象 + 设新的 `hostingView.rootView`，SwiftUI 检测到 View identity 变化就重建 view 树。

## Status 三态

| Status | 含义 | 行为 |
|---|---|---|
| `.installed` | 源+目标都已在本地 | 直接 `translate` |
| `.supported` | 支持但未下载 | 需要引导用户去系统设置下载 |
| `.unsupported` | 完全不支持 | 报错给用户 |

## 触发下载

framework **不会自动下载**语种包。你必须引导用户：

```swift
// 打开系统设置 → 语言与地区 → 翻译语言
if let url = URL(string: "x-apple.systempreferences:com.apple.preference.language") {
    NSWorkspace.shared.open(url)
}
```

或者让用户去：
**系统设置 → 通用 → 语言与地区 → 翻译语言 → 点 + 添加**

下载量：每个语种包几百 MB。

## 错误处理

`TranslationError` 几种（macOS 15.0+ base case）：

```swift
do {
    let response = try await session.translate(text)
} catch let error as TranslationError {
    switch error {
    case .unsupportedSourceLanguage: ...
    case .unsupportedTargetLanguage: ...
    case .unableToIdentifyLanguage: ...
    case .nothingToTranslate: ...
    case .internalError: ...
    default: ...  // macOS 26+ 的 .notInstalled / .alreadyCancelled 在这里
    }
}
```

**坑 1**：`TranslationError` 没有 `==` 操作符，但有 `~=`：
```swift
TranslationError.notInstalled ~= error   // pattern match 可用
error == .notInstalled                    // 编译错
```

**坑 2（AppKit 桥接特有）**：因为 bridge 里只能存 `String(describing: error)`，pattern matching 没法用，只能用 `errString.contains("unsupportedSourceLanguage")` 之类的关键字匹配。

## 与其它翻译方案对比

| 方案 | 联网 | 费用 | 延迟 | 隐私 |
|---|---|---|---|---|
| Translation framework | 否 | 免费 | 极低（本地）| 完全本地 |
| Google Translate API | 是 | 按字符付费 | 网络 RTT | 数据上云 |
| DeepL API | 是 | 订阅 | 网络 RTT | 数据上云 |
| 第三方 ML 模型（如 NLLB）| 否 | 自建 | 视模型 | 完全本地 |

## SwiftUI 一行代码方案（仅 SwiftUI 项目）

```swift
import Translation
import SwiftUI

struct ContentView: View {
    @State private var showTranslation = false
    @State private var text = "Hello, World!"

    var body: some View {
        Text(text)
            .translationPresentation(isPresented: $showTranslation, text: text)
    }
}
```

`.translationPresentation` 自动弹系统翻译面板。**SwiftUI 项目才能用**，AppKit 项目得自己桥接（见上）。

## API 限制 / 注意事项

1. **macOS 15.0+ / iOS 18.0+ 才可用**——base API
2. **`TranslationSession(source:target:)` 不存在**——那个 init 在 `Configuration` 上
3. **`TranslationSession(installedSource:target:)` 是 macOS 26.0+**——非 SwiftUI 直接构造的便捷路径
4. **`.notInstalled` 错误码是 macOS 26.0+**——之前版本抛其他错误
5. **不能在 `Daemon` / 后台 Service 里跑**——需要用户上下文
6. **每次翻译都创建新 session**——session 不保持状态
7. **没有进度回调**——只有「翻译中...」→「完成」两态
8. **AppKit 没有翻译面板 View**——必须自己组装 UI + 用 NSHostingView 桥接
9. **`Locale.Language(identifier:)`** 是 Foundation 标准（不是 Translation 专属）
10. **纯 CLT 的 swiftc 不能用 SwiftUI 编译器宏**（`@State` / `@Observable`）——必须 Xcode，或者绕过

## 实战建议

1. **先查 LanguageAvailability 再 translate**——避免 `notInstalled` 错误抛给用户
2. **入口语种对选择 UI**——用户不会知道哪个语种已下载
3. **首次未下载时给清晰引导**——给按钮跳系统设置
4. **批量翻译用 `translate(batch:)` 返回 `BatchResponse`（AsyncSequence）**——单条用 `translate(_:)`
5. **`AttributedString` 版本需要 macOS 26.4+**——保留原文样式的翻译，富文本场景才需要
6. **AppKit + Translation 用 NSHostingView 桥接**——macOS 15.0 唯一路径
7. **不要在纯 CLT 编译里用 `@State` / `@Observable`**——用重建 rootView + 普通 class 绕开

## 参考

- Apple: [Translation framework docs](https://developer.apple.com/documentation/translation)
- 本机 SDK 头文件：`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Translation.framework`
- Swift 接口：`Versions/A/Modules/Translation.swiftmodule/arm64e-apple-macos.swiftinterface`
- [Translating text within your app](https://developer.apple.com/documentation/translation/translating-text-within-your-app)

## 在本项目里的应用

- `HelloWorld.swift` 的 `TranslationView` 用 NSHostingView 嵌 `TranslationBridgeView`
- `LSMinimumSystemVersion` = 15.0
- `build.sh` 用 `-framework Cocoa -framework Translation -framework SwiftUI`
- 桥接对象 `TranslationBridge` 是普通 class，每次翻译新建一个
- `hostingView.rootView` 重新赋值触发 SwiftUI 重建 → `.translationTask` 重新执行
- Timer 50ms 轮询 bridge 的 `lastResponse` / `lastError`，结果回写到 AppKit UI
