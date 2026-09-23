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

## AppKit 项目怎么拿 session：NSHostingView + continuation 桥接

因为 `.translationTask` 是 SwiftUI 专属，AppKit 项目必须桥接。**当前架构是单向 request/response + CheckedContinuation**：

```
AppKit Controller                          SwiftUI View
       │                                          │
       │ await bridge.requestTranslate(...)        │
       │───────────────────────────────────────>   │
       │     (更新 pendingConfiguration,           │
       │      generation += 1,                     │
       │      装好 CheckedContinuation)            │
       │                                          │ body 重建（.id(generation)）
       │                                          │ .translationTask 闭包启动
       │                                          │ session 在手里
       │                                          │ bridge.deliverSession(session)
       │ <─────────────────────────────────────  │  (resume continuation)
       │  (session 拿到了)                          │
       │  try await session.translate(text)        │
       │  return result                            │
       ▼
  更新 AppKit UI
```

**优势**（对比旧的"Timer 50ms 轮询 + 共享可变属性"）：
- ✅ 没有 race condition：`@MainActor` 强制所有访问在主线程
- ✅ 没有 Timer 生命周期管理：`await` 自然结束
- ✅ 状态机清晰：request / response 单向流
- ✅ 没有"什么时候算翻译完成"的歧义：continuation resume 即完成

### 完整代码（本项目当前实现）

```swift
// 1. 桥接对象：@MainActor actor-style，单向 request/response
@available(macOS 15.0, *)
@MainActor
final class TranslationBridge {
    private(set) var generation: Int = 0
    private(set) var pendingConfiguration: TranslationSession.Configuration?
    private(set) var pendingText: String = ""

    private var sessionContinuation: CheckedContinuation<TranslationSession, Error>?
    private var cachedSession: TranslationSession?

    /// AppKit 调用入口：一次 async 调用拿到翻译结果
    func requestTranslate(text: String, configuration: TranslationSession.Configuration) async throws -> String {
        // 1. 更新 pending 状态 + generation（让 SwiftUI view 知道要重建）
        self.pendingText = text
        self.pendingConfiguration = configuration
        self.generation &+= 1
        self.cachedSession = nil

        // 2. 等待 SwiftUI 通过 translationTask 闭包把 session 推过来；带超时
        let session = try await fetchSessionWithTimeout()

        // 3. 调 session.translate，返回结果
        let response = try await session.translate(text)
        return response.targetText
    }

    private func fetchSessionWithTimeout() async throws -> TranslationSession {
        if let s = cachedSession { return s }
        return try await withThrowingTaskGroup(of: TranslationSession.self) { group in
            group.addTask { try await self.waitForSession() }
            group.addTask {
                try await Task.sleep(nanoseconds: requestTimeoutSeconds * 1_000_000_000)
                throw TranslationError.internalError
            }
            defer { group.cancelAll() }
            for try await result in group {
                return result
            }
            throw TranslationError.internalError
        }
    }

    private func waitForSession() async throws -> TranslationSession {
        try await withCheckedThrowingContinuation { cont in
            self.sessionContinuation = cont
        }
    }

    /// SwiftUI view 调用：把新 session 推回来 resume continuation
    func deliverSession(_ session: TranslationSession) {
        self.cachedSession = session
        sessionContinuation?.resume(returning: session)
        sessionContinuation = nil
    }

    func cancelPending(error: Error) {
        sessionContinuation?.resume(throwing: error)
        sessionContinuation = nil
    }
}

// 2. SwiftUI 容器：把 session 推回 bridge
@available(macOS 15.0, *)
struct TranslationBridgeView: View {
    let bridge: TranslationBridge

    var body: some View {
        EmptyView()
            .id(bridge.generation)   // generation 变化 → view identity 变化 → body 重执行
            .translationTask(bridge.pendingConfiguration) { session in
                bridge.deliverSession(session)   // 把 session 推回 bridge，resume continuation
            }
    }
}

// 3. AppKit 端：一次 await 拿结果，每次翻译销毁重建 hosting view
@available(macOS 15.0, *)
final class TranslationView: NSView {
    private let bridge = TranslationBridge()
    private var bridgeHostingView: NSHostingView<TranslationBridgeView>?

    @objc private func translateClicked() {
        // ...
        Task { @MainActor in
            // ...语种预检查...
            self.installBridgeHostingView()  // 销毁旧 hosting view + addSubview 新的
            do {
                let result = try await self.bridge.requestTranslate(
                    text: inputText,
                    configuration: TranslationSession.Configuration(source: ..., target: ...)
                )
                self.outputView.string = result
            } catch {
                // 错误处理
            }
        }
    }

    private func installBridgeHostingView() {
        if let old = bridgeHostingView {
            old.removeFromSuperview()
            bridgeHostingView = nil
        }
        let hosting = NSHostingView(rootView: TranslationBridgeView(bridge: bridge))
        hosting.frame = .zero
        addSubview(hosting)
        bridgeHostingView = hosting
    }
}
```

### 三个关键设计决策

1. **`@MainActor` 替代锁**：所有访问都在主线程，不需要 `NSLock` / `os_unfair_lock`。Swift 编译器静态保证安全。

2. **`.id(bridge.generation)` + 销毁重建 hosting view 双保险**：
   - `.id(Int)` 让 SwiftUI 看到 view identity 变化 → body 重执行 → `.translationTask` 重新调用闭包
   - 销毁旧 `NSHostingView` + addSubview 新的 = 100% 触发 SwiftUI 重新构造整个 hosting 容器
   - 实测两者单独都偶尔失灵，组合起来最稳

3. **`await` 而不是回调或共享属性**：
   - AppKit 调用 `await bridge.requestTranslate(...)` 直接拿到结果字符串
   - SwiftUI 闭包内 `bridge.deliverSession(session)` 只做一件事：resume continuation
   - 没有任何"轮询 lastResponse"的代码

### 为什么不用 @Observable / @State

纯 CLT（Command Line Tools）的 swiftc **不能加载** SwiftUI 的 macros plugin——必须有 Xcode。`@Observable` / `@State` 等 SwiftUI 编译器宏因此无法用，编译报错：

```
error: external macro implementation type 'SwiftUIMacros.StateMacro' could not be found
```

绕过方案：用**普通 class** 当桥接对象；用 `.id(Int)` 强制 view 重建；用 `await` continuation 通信。

### 如果有 Xcode，可以更优雅

如果用 Xcode + Swift Package Manager，编译路径会自动加载 SwiftUI 编译器宏。这时可以用：

```swift
@Observable
final class TranslationBridge {
    var pendingConfiguration: TranslationSession.Configuration?
    var pendingText: String = ""
    // ... 其他字段
}

struct TranslationBridgeView: View {
    @Bindable var bridge: TranslationBridge  // @Observable 自动监听
    var body: some View {
        // SwiftUI 自动监听 bridge 字段变化重建 view
        // 不需要 .id()，不需要 CheckedContinuation
    }
}
```

这条路本项目走不了（纯 CLT），所以用上面的 `@MainActor + continuation` 方案达到等价效果。

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
11. **`.translationTask` 闭包不自动重启**——修改 `NSHostingView.rootView` 不会触发，需要 `.id()` 或销毁重建
12. **`LanguageAvailability.status()` 状态可能不稳定**——首次检查是 `.supported` 的语种对，短时间后再查可能变成 `.installed`（系统可能在我们看不见的层面触发了下载）

## 实战建议

1. **先查 LanguageAvailability 再 translate**——避免 `notInstalled` 错误抛给用户
2. **入口语种对选择 UI**——用户不会知道哪个语种已下载
3. **首次未下载时给清晰引导**——给按钮跳系统设置
4. **批量翻译用 `translate(batch:)` 返回 `BatchResponse`（AsyncSequence）**——单条用 `translate(_:)`
5. **`AttributedString` 版本需要 macOS 26.4+**——保留原文样式的翻译，富文本场景才需要
6. **AppKit + Translation 用 NSHostingView 桥接**——macOS 15.0 唯一路径
7. **不要在纯 CLT 编译里用 `@State` / `@Observable`**——用普通 class + 销毁重建 hosting view 绕开
8. **不要复用 `NSHostingView`**——每次翻译都 `removeFromSuperview() + addSubview(new)`，否则 `.translationTask` 闭包可能不重启
9. **加日志看闭包是否启动**——长文本卡住时如果 AppKit 端一直在轮询但 SwiftUI 闭包没进入，就是 hosting view 复用问题
10. **状态判定不要只看一次**——首次 `.supported` 的语种对可能很快变成 `.installed`，因为 `.translationTask` 配置本身可能触发系统下载

## 参考

- Apple: [Translation framework docs](https://developer.apple.com/documentation/translation)
- 本机 SDK 头文件：`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Translation.framework`
- Swift 接口：`Versions/A/Modules/Translation.swiftmodule/arm64e-apple-macos.swiftinterface`
- [Translating text within your app](https://developer.apple.com/documentation/translation/translating-text-within-your-app)

## 在本项目里的应用

- `HelloWorld.swift` 的 `TranslationView` 用 NSHostingView 嵌 `TranslationBridgeView`
- `LSMinimumSystemVersion` = 15.0
- `build.sh` 用 `-framework Cocoa -framework Translation -framework SwiftUI`
- 桥接对象 `TranslationBridge` 是普通 class，每次翻译新建一个（含 UUID requestID）
- `TranslationBridgeView` 用 `.id(bridge.requestID)` 强制 view identity 变化
- **每次翻译都 `removeFromSuperview() + addSubview(new)`** 销毁旧 hosting view——这是修复长文本卡死的关键
- Timer 50ms 轮询 bridge 的 `lastResponse` / `lastError`，结果回写到 AppKit UI
- 加了日志 Tab（⌘, 打开设置可关掉日志写入）和 `LogStore.enabled` 开关
- `setStatus(...)` 同时调 `LogStore.shared.append(...)` 写日志，方便后续诊断
- 窗口 UI 三段（Hello / 翻译 / 日志）固定不变，偏好只控制日志写入开关
