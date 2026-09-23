# Translation 调试日志：session.translate 卡死的根因分析

> 适用：macOS 15+ Translation framework 桥接到 AppKit 的场景
> 调试周期：2026-09-22 ~ 2026-09-23
> 最终方案：polling cachedSession（替换 CheckedContinuation）

## 背景

项目 `031-xcode-cmd-test/` 把 macOS 15+ Translation framework 通过 `NSHostingView` 嵌入 AppKit 窗口。架构：

```
AppKit TranslationView
       │ await bridge.requestTranslate(...)
       ▼
@MainActor TranslationBridge
       │ generation += 1，pendingConfiguration 更新
       ▼
NSHostingView(TranslationBridgeView)
       │ .translationTask(configuration) { session in bridge.deliverSession(session) }
       ▼
TranslationSession.translate(text)
```

目标是用 `async/await` continuation 模式做单向 request/response，避免 Timer 轮询。

## 症状

点翻译按钮后，约 50% 概率卡 30 秒超时：

```
[Bridge] → bridge.requestTranslate(text 长度=59, gen=0)
[SwiftUI] → .translationTask 闭包启动（generation=1，...）
[Bridge] ← deliverSession（generation=1，session.source=Optional("en")）
[Bridge] ✗ requestTranslate 超时（30s）  ← 30 秒后才报
```

**关键观察**：`deliverSession` 已经 log 出来了（说明 SwiftUI 闭包已经调过 `bridge.deliverSession(session)`），但 30 秒后仍然超时。

## 调试过程：六次失败的尝试

### 尝试 1：用 `for try await result in group` 等两个 task

```swift
group.addTask { try await self.waitForSession() }
group.addTask {
    try await Task.sleep(nanoseconds: 30_000_000_000)
    throw TranslationError.internalError
}
for try await result in group {
    return result
}
```

**结果**：间歇性卡死。`for try await` 在两个 task **同时完成**时迭代顺序不确定——可能先迭代到 sleep 抛的 `internalError`，导致已 resume 的 session 被丢弃。

### 尝试 2：把超时缩到更短、改 retry 间隔

`session.translate` 加重试机制（attempt 1 失败 → 等 0.3s → attempt 2）。但**同一个 session** 第二次调大概率仍卡——retry 解决不了根因。

### 尝试 3：每次销毁重建 NSHostingView + 新 TranslationBridge

之前已有"销毁重建 hosting view"修长文本卡死的 bug。这次重提是想隔离 session 复用。但**仍卡**。

### 尝试 4：调 `session.prepareTranslation()` 让 framework 准备好

Apple 官方 API（macOS 15+）。但**这个 API 自己也卡**——证实问题在 Translation framework 内部。

### 尝试 5：`LanguageAvailability.status()` 重试 5 次（每次 0.5s）

修另一个症状：首次翻译显示"语言包未下载"，第二次就好。猜测是 macOS 系统层在后台主动下载语种包。重试给系统机会"准备就绪"。**这条修复有效**（保留），但和卡死 bug 是两个独立问题。

### 尝试 6：用 `group.next()` 替代 `for try await`

```swift
for _ in 0..<2 {
    do {
        if let result = try await group.next() { return result }
    } catch {
        // 忽略 task 抛的错
    }
}
```

**结果**：仍然卡死，但日志揭出新信息：

```
[Bridge] ✗ requestTranslate 超时（30s）
[Bridge]   fetchSession: 跳过 task 异常（internalError），取下一个
```

`group.next()` 的 catch 跑了，但**没拿到 session**。说明 `waitForSession` 这个 task 30 秒内根本没 resume 过。

## 真正的根因（第七次才看清）

`sessionContinuation` 的 **check-then-register race**：

```swift
// 1. 旧 fetchSessionWithTimeout
group.addTask { try await self.waitForSession() }
// ↑ waitForSession 任务加入队列，但**不一定立即被调度**

// 2. 同时 deliverSession 在 SwiftUI 闭包内被调用
func deliverSession(_ session: TranslationSession) {
    self.cachedSession = session
    sessionContinuation?.resume(returning: session)  // ← sessionContinuation 可能仍是 nil！
    sessionContinuation = nil
}
```

`withCheckedThrowingContinuation { cont in sessionContinuation = cont }` 看似同步，但**前提是 waitForSession 函数本身被调度执行**。SwiftUI 的 `.translationTask` 闭包在主线程同步触发 `deliverSession`——可能**早于** waitForSession 任务被调度到执行队列。

时序竞争：

```
fetchSessionWithTimeout 主线程:
  group.addTask(waitForSession)   ← 加入队列
  group.addTask(sleep)            ← 加入队列
  group.next()                    ← 主线程挂起，调度器开始选 task 运行

调度器（不可预测）:
  [可能路径 A] waitForSession 先跑 → 设 cont → 挂起等待 resume
  [可能路径 B] sleep 先跑 → 30s 后抛错
  [可能路径 C] deliverSession 先跑（在 SwiftUI 闭包内，主线程触发）→ sessionContinuation 是 nil → resume 跳过
```

**路径 C 一旦发生**：session 写入 `cachedSession` 但 continuation 路径丢失。30 秒后 waitForSession 才被调度、设 cont、永远等不到 resume——超时。

**本质上**：continuation 模型 + check-then-register（先查 cached、再注册 waiter）天生有 race。修这个 race 要加锁或更复杂的同步，反而比直接 polling 更难写对。

## 修正：状态模型 vs 事件模型

回顾整个调试，正确的理解是：

> 「等待 session ready」是一种**状态**而非**事件**。callback 写入 `cachedSession` 是状态变更；waiter 等状态进入 ready 是订阅状态。polling 是订阅状态最简单的实现，**避免**了 check-then-register race。

不是「continuation 不可靠」——而是「这个场景语义上是状态而非事件」。

### 关键设计：单一同步原语

```swift
@MainActor
final class TranslationBridge {
    private var cachedSession: TranslationSession?

    func deliverSession(_ session: TranslationSession) {
        cachedSession = session   // ← 唯一同步点
    }
}
```

不要在同一处混用 `cachedSession + continuation + Task.sleep` 三种同步机制——会增加复杂度，不解决问题。

### `try await Task.sleep` 的关键细节

之前用 `try? await Task.sleep(...)` —— `try?` 把 `CancellationError` 吞掉了。即使外层 `Task` 被 `cancel()`，`fetchSessionWithTimeout` 内的 polling 循环也会跑完整个 30 秒超时。

改用：

```swift
let deadline = ContinuousClock.now + .seconds(30)
while ContinuousClock.now < deadline {
    try Task.checkCancellation()  // ← 让 cancellation 立刻传上来
    if let session = cachedSession { return session }
    try await Task.sleep(for: .milliseconds(50))  // ← 不吞 cancellation
}
throw TranslationError.internalError
```

`ContinuousClock` 也比 `Date()` 准——不受系统时间调整影响。

## 最终修复：polling cachedSession

抛弃 continuation 机制，改用最朴素的轮询：

```swift
private func fetchSessionWithTimeout() async throws -> TranslationSession {
    let startTime = Date()
    let timeoutSeconds: TimeInterval = 30
    while Date().timeIntervalSince(startTime) < timeoutSeconds {
        if let s = cachedSession { return s }
        try? await Task.sleep(nanoseconds: 50_000_000)
    }
    throw TranslationError.internalError
}
```

`deliverSession` 仍然写入 `cachedSession`（同步、立即），不再依赖 continuation：

```swift
func deliverSession(_ session: TranslationSession) {
    self.cachedSession = session
    sessionContinuation?.resume(returning: session)  // 可选，已无意义但保留无害
    sessionContinuation = nil
}
```

**为什么 50ms polling 可接受**：
- `cachedSession` 是**同步字段**，deliverSession 一写入，polling 第一次循环（0-50ms 后）就能拿到
- 实际只 sleep 0-50ms，不是真正"轮询"
- 100% 可靠，不依赖任何调度顺序
- 30 秒兜底防止永久挂起

## 为什么一开始不直接 polling

最初设计目标是「替代 Timer 50ms 轮询」（用户评价建议）。continuation 模式理论上是"正确"的 Swift concurrency 风格——单向 request/response，无 race condition，无 Timer 生命周期。

但实践上：
- `withCheckedThrowingContinuation` 在跨 actor 边界的调度时序**不可预测**
- AppKit ↔ SwiftUI 跨边界的"主线程同步触发"和"TaskGroup addTask 调度"之间的竞争**理论存在**
- `await bridge.requestTranslate(...)` 这条干净的单向流，被 framework 内部的隐性依赖（continuation 必须被同一个调度器认领）破坏了

教训：**async/await continuation 不是万能解药**——当 callback 可能在 await 之前到达时（如跨 actor / 跨框架边界），需要 polling 这种"显式等待状态变化"的笨办法兜底。

## 经验总结

### 关于 Translation framework（macOS 15+）

1. **`.translationTask` 闭包给出的 session 几乎立刻可用**——但仍有 ~50% 概率调 `session.translate(...)` 卡死（framework 内部 bug，路径不清晰）
2. **`session.prepareTranslation()` 也会卡**——Apple 官方"等准备好"API 自身在间歇性场景下也不可靠
3. **`LanguageAvailability.status()` 状态不稳定**——首次检查 `.supported` 6 秒后再查可能变 `.installed`（macOS 系统层在后台主动下载语种包）。**必须重试**

### 关于 async/await + continuation 模式

1. **跨 actor 边界**（AppKit 主线程 ↔ SwiftUI 闭包）用 `CheckedContinuation` 容易出调度竞争
2. **polling 字段**虽然朴素，但**完全可预测**，是兜底跨边界问题的可靠手段
3. **TaskGroup 内的 sleep task 抛错**和**真正的业务 task 完成**——顺序不可预测，必须用 `group.next()` 显式控制
4. **`for try await result in group`** 在多 task 同时完成时是 race 的源头，能不用就不用

### 关于 NSHostingView + SwiftUI 在纯 CLT 下

1. 纯 CLT 的 swiftc **不能用 `@Observable` / `@State`**（缺 SwiftUIMacros 插件）——必须 Xcode
2. **`NSHostingView.rootView = ...` 赋值不一定触发 SwiftUI view tree 重建**——必须 `removeFromSuperview() + addSubview(new)` 销毁重建
3. **`.id(...)` modifier** 是另一条触发 view identity 变化的路径，但实测**也不总是稳**——销毁重建是最稳的

### 调试方法

1. **每个关键点加日志**——`LogStore.shared.append(...)` + 「日志」Tab 直接看
2. **时序竞争用日志定位**：打点足够细，能区分"代码没执行"vs"代码执行了但被 race 吞掉"
3. **`@unchecked Sendable` + `NSLock`** 适合共享日志存储；`@MainActor` 适合业务状态机

## 最终代码结构（关键部分）

```swift
@MainActor
final class TranslationBridge {
    private(set) var generation: Int = 0
    private(set) var pendingConfiguration: TranslationSession.Configuration?
    private(set) var pendingText: String = ""

    // 单一同步原语：ready state。deliverSession 写入，fetchSessionWithTimeout polling 读。
    private var cachedSession: TranslationSession?

    func requestTranslate(text: String, configuration: Configuration) async throws -> String {
        self.pendingText = text
        self.pendingConfiguration = configuration
        self.generation &+= 1
        self.cachedSession = nil

        let session = try await fetchSessionWithTimeout()
        try? await Task.sleep(nanoseconds: 200_000_000)  // 200ms 给 framework warmup

        // 重试 2 次，间隔 1.5s
        var attempt = 0
        let maxAttempts = 2
        while true {
            attempt += 1
            do {
                let response = try await session.translate(text)
                return response.targetText
            } catch {
                if attempt >= maxAttempts { throw error }
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
    }

    private func fetchSessionWithTimeout() async throws -> TranslationSession {
        // 状态轮询：避免 check-then-register race。
        // try await（不是 try?）让 Task.cancel() 立刻传播；
        // ContinuousClock 比 Date() 准。
        let deadline = ContinuousClock.now + .seconds(30)
        while ContinuousClock.now < deadline {
            try Task.checkCancellation()
            if let session = cachedSession { return session }
            try await Task.sleep(for: .milliseconds(50))
        }
        throw TranslationError.internalError
    }

    /// SwiftUI view 调用：把新 session 写入 ready state
    func deliverSession(_ session: TranslationSession) {
        self.cachedSession = session
    }
}
```

AppKit 端：

```swift
@objc private func translateClicked() {
    Task { @MainActor in
        // 语种预检查（带 5 次 0.5s 重试，给 macOS 系统层后台下载时间）
        ...
        self.installBridgeHostingView()  // 销毁旧 hosting view + addSubview 新的

        do {
            let config = TranslationSession.Configuration(source: ..., target: ...)
            let result = try await self.bridge!.requestTranslate(text: inputText, configuration: config)
            self.outputView.string = result
        } catch { ... }
    }
}

private func installBridgeHostingView() {
    if let old = bridgeHostingView {
        old.removeFromSuperview()
        bridgeHostingView = nil
    }
    let newBridge = TranslationBridge()
    self.bridge = newBridge
    let hosting = NSHostingView(rootView: TranslationBridgeView(bridge: newBridge))
    hosting.frame = .zero
    addSubview(hosting)
    bridgeHostingView = hosting
}
```

## 相关文件

- `HelloWorld.swift` — 完整实现
- `docs/translation-framework.md` — Translation framework API 速览
- `docs/bundle-vs-single.md` — Bundle 化讨论
