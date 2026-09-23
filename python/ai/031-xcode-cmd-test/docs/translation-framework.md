# Translation framework 学习笔记

> 适用：macOS 26.0+ / iOS 18.0+。本机 SDK 27.0，本项目 target 锁 26.0。

## 它是什么

Apple 在 macOS 15 / iOS 18 引入的**系统级本地化翻译 framework**。不同于 Google Translate / DeepL 那种云端 API，它是**离线**的——首次用某语种对时下载本地翻译模型，之后无需网络。

## 能做什么

- 把任意文本翻译到目标语种
- 支持 100+ 语种
- 自动识别语种对支持状态（已下载 / 支持但未下载 / 不支持）
- `AttributedString` 翻译（macOS 26.4+），保留样式
- 批量翻译（async sequence）
- 与系统右键菜单的「翻译」共享同一个翻译引擎

## 核心 API

```swift
import Translation

// 1. 检查语种对支持
let availability = LanguageAvailability()
let status = await availability.status(
    from: Locale.Language(identifier: "en"),
    to: Locale.Language(identifier: "zh-Hans")
)
// status: .installed / .supported / .unsupported

// 2. 创建翻译会话（macOS 26+ convenience init，要求源语种已下载）
let session = TranslationSession(
    installedSource: Locale.Language(identifier: "en"),
    target: Locale.Language(identifier: "zh-Hans")
)

// 3. 翻译
let response = try await session.translate("Hello, World!")
print(response.targetText)  // "你好，世界！"
print(response.sourceLanguage)  // Locale.Language(en)
print(response.targetLanguage)  // Locale.Language(zh-Hans)
```

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

`TranslationError` 几种：

```swift
do {
    let response = try await session.translate(text)
} catch let error as TranslationError {
    switch error {
    case .unsupportedSourceLanguage: ...
    case .unsupportedTargetLanguage: ...
    case .unableToIdentifyLanguage: ...
    case .nothingToTranslate: ...
    case .notInstalled: ...     // macOS 26.0+
    case .alreadyCancelled: ... // macOS 26.0+
    case .internalError: ...
    default: ...
    }
}
```

**坑**：`TranslationError` 没有 `==` 操作符，但有 `~=`：
```swift
TranslationError.notInstalled ~= error   // pattern match 可用
error == .notInstalled                    // 编译错
```

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

`.translationPresentation` 自动弹系统翻译面板。但 SwiftUI 项目才能用，**AppKit 项目必须自己画 UI**，这就是为什么本项目用 `NSSegmentedControl` + `NSPopUpButton` + `NSTextView` 自己组装。

## API 限制 / 注意事项

1. **macOS 15.0+ / iOS 18.0+ 才可用**——base API
2. **`installedSource:target:` 是 macOS 26.0+ convenience init**——base `init(source:target:)` 在某些 SDK 上解析不到默认参数，需要明确写
3. **`.notInstalled` 是 macOS 26.0+**——之前版本没有这个错误码
4. **不能在 `Daemon` / 后台 Service 里跑**——需要用户上下文
5. **每次翻译都创建新 session**——session 不保持状态
6. **没有进度回调**——只有「翻译中...」→「完成」两态
7. **AppKit 没有翻译面板 View**——必须自己组装 UI
8. **`Locale.Language(identifier:)`** 是 Foundation 标准（不是 Translation 专属）

## 实战建议

1. **先查 LanguageAvailability 再 translate**——避免 `notInstalled` 错误抛给用户
2. **入口语种对选择 UI**——用户不会知道哪个语种已下载
3. **首次未下载时给清晰引导**——给按钮跳系统设置
4. **批量翻译用 `translate(batch:)` 返回 `BatchResponse`（AsyncSequence）**——单条用 `translate(_:)`
5. **`AttributedString` 版本需要 macOS 26.4+**——保留原文样式的翻译，富文本场景才需要

## 参考

- Apple: [Translation framework docs](https://developer.apple.com/documentation/translation)
- 本机 SDK 头文件：`/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Translation.framework`
- Swift 接口：`Versions/A/Modules/Translation.swiftmodule/arm64e-apple-macos.swiftinterface`

## 在本项目里的应用

- `HelloWorld.swift` 的 `TranslationView` 类封装了完整流程
- `LSMinimumSystemVersion` = 26.0（与编译 target 对齐）
- `build.sh` 加 `-framework Translation`
- 窗口默认 720×520，加 segmented control 切 Hello / 翻译两个 Tab
