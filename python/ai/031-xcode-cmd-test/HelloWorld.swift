import Cocoa
import SwiftUI
import Translation

// 顶层常量：TranslationBridge 用的超时秒数（非 main actor 隔离）
fileprivate let requestTimeoutSeconds: UInt64 = 30

// 启动时读 SDK 与 CLT 路径，作为窗口副标题展示
func shellRead(_ path: String, _ args: [String]) -> String {
    let task = Process()
    task.executableURL = URL(fileURLWithPath: path)
    task.arguments = args
    let pipe = Pipe()
    let errPipe = Pipe()
    task.standardOutput = pipe
    task.standardError = errPipe
    do {
        try task.run()
    } catch {
        return "unknown"
    }
    task.waitUntilExit()
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    return String(data: data, encoding: .utf8)?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? "unknown"
}

let sdkVersion = shellRead("/usr/bin/xcrun", ["--show-sdk-version", "--sdk", "macosx"])
let cltPath = shellRead("/usr/bin/xcode-select", ["-p"])

// MARK: - 语种预设

struct LanguageChoice {
    let display: String
    let identifier: String
    var language: Locale.Language { Locale.Language(identifier: identifier) }
}

let sourceLanguages: [LanguageChoice] = [
    LanguageChoice(display: "自动 (跟随系统)", identifier: "auto"),
    LanguageChoice(display: "English", identifier: "en"),
    LanguageChoice(display: "中文 (简体)", identifier: "zh-Hans"),
    LanguageChoice(display: "日文", identifier: "ja"),
    LanguageChoice(display: "法文", identifier: "fr"),
    LanguageChoice(display: "德文", identifier: "de"),
    LanguageChoice(display: "韩文", identifier: "ko"),
    LanguageChoice(display: "西班牙文", identifier: "es"),
]

let targetLanguages: [LanguageChoice] = [
    LanguageChoice(display: "English", identifier: "en"),
    LanguageChoice(display: "中文 (简体)", identifier: "zh-Hans"),
    LanguageChoice(display: "中文 (繁体)", identifier: "zh-Hant"),
    LanguageChoice(display: "日文", identifier: "ja"),
    LanguageChoice(display: "法文", identifier: "fr"),
    LanguageChoice(display: "德文", identifier: "de"),
    LanguageChoice(display: "韩文", identifier: "ko"),
    LanguageChoice(display: "西班牙文", identifier: "es"),
]

// MARK: - HelloView (Tab 1)

final class HelloView: NSView {
    init() {
        super.init(frame: .zero)
        translatesAutoresizingMaskIntoConstraints = false

        let bundleId = Bundle.main.bundleIdentifier ?? "(no bundle ID)"
        let resourcesPath = Bundle.main.resourcePath ?? "(no resourcePath)"

        let titleLabel = NSTextField(labelWithString: "Hello, World!")
        titleLabel.font = NSFont.systemFont(ofSize: 28, weight: .bold)
        titleLabel.textColor = .labelColor
        titleLabel.alignment = .center
        titleLabel.translatesAutoresizingMaskIntoConstraints = false

        let subtitleLabel = NSTextField(labelWithString: "Xcode CLT 检测通过 ✓")
        subtitleLabel.font = NSFont.systemFont(ofSize: 14)
        subtitleLabel.textColor = .systemGreen
        subtitleLabel.alignment = .center
        subtitleLabel.translatesAutoresizingMaskIntoConstraints = false

        let sdkLabel = NSTextField(labelWithString: "macOS SDK: \(sdkVersion)")
        sdkLabel.font = NSFont.systemFont(ofSize: 11)
        sdkLabel.textColor = .secondaryLabelColor
        sdkLabel.alignment = .center
        sdkLabel.translatesAutoresizingMaskIntoConstraints = false

        let pathLabel = NSTextField(labelWithString: "CLT: \(cltPath)")
        pathLabel.font = NSFont.systemFont(ofSize: 10)
        pathLabel.textColor = .tertiaryLabelColor
        pathLabel.alignment = .center
        pathLabel.lineBreakMode = .byTruncatingMiddle
        pathLabel.translatesAutoresizingMaskIntoConstraints = false

        let bundleLabel = NSTextField(labelWithString: "Bundle ID: \(bundleId)")
        bundleLabel.font = NSFont.systemFont(ofSize: 11, weight: .medium)
        bundleLabel.textColor = .systemBlue
        bundleLabel.alignment = .center
        bundleLabel.translatesAutoresizingMaskIntoConstraints = false

        let resourcesLabel = NSTextField(labelWithString: "Resources: \(resourcesPath)")
        resourcesLabel.font = NSFont.systemFont(ofSize: 9)
        resourcesLabel.textColor = .tertiaryLabelColor
        resourcesLabel.alignment = .center
        resourcesLabel.lineBreakMode = .byTruncatingMiddle
        resourcesLabel.translatesAutoresizingMaskIntoConstraints = false

        addSubview(titleLabel)
        addSubview(subtitleLabel)
        addSubview(bundleLabel)
        addSubview(sdkLabel)
        addSubview(pathLabel)
        addSubview(resourcesLabel)

        NSLayoutConstraint.activate([
            titleLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            titleLabel.topAnchor.constraint(equalTo: topAnchor, constant: 30),

            subtitleLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            subtitleLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 10),

            bundleLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            bundleLabel.topAnchor.constraint(equalTo: subtitleLabel.bottomAnchor, constant: 14),

            sdkLabel.centerXAnchor.constraint(equalTo: centerXAnchor),
            sdkLabel.topAnchor.constraint(equalTo: bundleLabel.bottomAnchor, constant: 10),

            pathLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            pathLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            pathLabel.topAnchor.constraint(equalTo: sdkLabel.bottomAnchor, constant: 4),

            resourcesLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            resourcesLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            resourcesLabel.topAnchor.constraint(equalTo: pathLabel.bottomAnchor, constant: 2),
        ])
    }

    required init?(coder: NSCoder) { fatalError() }
}

// MARK: - TranslationView (Tab 2)

// MARK: - TranslationBridge（@MainActor 单向 request/response + continuation）

@available(macOS 15.0, *)
@MainActor
final class TranslationBridge {
    // 每次翻译 +1，TranslationBridgeView 用 .id(bridge.generation) 触发 view identity 变化
    private(set) var generation: Int = 0

    // 最新 pending 请求的 configuration —— SwiftUI .translationTask 读这个
    private(set) var pendingConfiguration: TranslationSession.Configuration?
    private(set) var pendingText: String = ""

    // SwiftUI 通过 deliverSession 把新 session 推回来；AppKit await 拿到
    private var sessionContinuation: CheckedContinuation<TranslationSession, Error>?
    private var cachedSession: TranslationSession?

    // 超时秒数（在文件顶层单独声明，避免 main actor 隔离冲突）


    /// AppKit 调用入口。一次 async 调用拿到翻译结果。
    func requestTranslate(text: String, configuration: TranslationSession.Configuration) async throws -> String {
        LogStore.shared.append("→ bridge.requestTranslate(text 长度=\(text.count), gen=\(generation))", source: "Bridge")

        // 1. 更新 pending 状态 + generation（让 SwiftUI view 知道要重建）
        self.pendingText = text
        self.pendingConfiguration = configuration
        self.generation &+= 1
        self.cachedSession = nil

        // 2. 等待 SwiftUI 通过 translationTask 闭包把 session 推过来；带超时
        let session = try await fetchSessionWithTimeout()

        // 3. 调 session.translate，带重试
        // 拿 session 后等 200ms（让 framework 完成内部 warmup），
        // 然后 attempt 1/2 重试（每次失败等 1.5s 再重试，给 framework 状态恢复时间）
        LogStore.shared.append(
            "→ session.translate 前等 200ms（让 framework 完成 warmup）",
            source: "Bridge"
        )
        try? await Task.sleep(nanoseconds: 200_000_000)

        var attempt = 0
        let maxAttempts = 2
        while true {
            attempt += 1
            LogStore.shared.append(
                "→ session.translate 开始（attempt \(attempt)/\(maxAttempts)，text 长度=\(text.count)）",
                source: "Bridge"
            )
            let translateStart = Date()
            do {
                let response = try await session.translate(text)
                let elapsed = String(format: "%.2f", Date().timeIntervalSince(translateStart))
                LogStore.shared.append(
                    "← session.translate 返回（attempt \(attempt)，耗时 \(elapsed)s，target 长度=\(response.targetText.count)）",
                    source: "Bridge"
                )
                return response.targetText
            } catch {
                let elapsed = String(format: "%.2f", Date().timeIntervalSince(translateStart))
                LogStore.shared.append(
                    "✗ session.translate 抛错（attempt \(attempt)，耗时 \(elapsed)s）：\(error)",
                    source: "Bridge"
                )
                if attempt >= maxAttempts {
                    throw error
                }
                // 重试前等 1.5s（让 Translation framework 内部状态真正恢复）
                LogStore.shared.append(
                    "  retry 前等 1.5s 让 framework 状态恢复",
                    source: "Bridge"
                )
                try? await Task.sleep(nanoseconds: 1_500_000_000)
            }
        }
    }

    private func fetchSessionWithTimeout() async throws -> TranslationSession {
        // 直接 polling cachedSession，每 50ms 检查一次
        // 原因：waitForSession 任务调度时机不确定，deliverSession 可能
        // 在 waitForSession 设置 sessionContinuation 之前就调用，
        // 导致 ?.resume 拿到 nil、session 丢失。
        // polling 直接读 cachedSession 字段（同步），不依赖 continuation 调度。
        let startTime = Date()
        let timeoutSeconds: TimeInterval = 30
        while Date().timeIntervalSince(startTime) < timeoutSeconds {
            if let s = cachedSession { return s }
            try? await Task.sleep(nanoseconds: 50_000_000)
        }
        LogStore.shared.append(
            "✗ fetchSession 超时（\(Int(timeoutSeconds))s，cachedSession 始终为 nil）",
            source: "Bridge"
        )
        throw TranslationError.internalError
    }

    /// SwiftUI view 调用：把新 session 推进来 resume continuation
    func deliverSession(_ session: TranslationSession) {
        LogStore.shared.append(
            "← deliverSession（generation=\(generation)，session.source=\(String(describing: session.sourceLanguage?.minimalIdentifier))）",
            source: "Bridge"
        )
        self.cachedSession = session
        sessionContinuation?.resume(returning: session)
        sessionContinuation = nil
    }

    /// 让 awaiter 抛错而不是永久阻塞（窗口关闭、hosting view 销毁时调用）
    func cancelPending(error: Error) {
        sessionContinuation?.resume(throwing: error)
        sessionContinuation = nil
    }
}

// MARK: - TranslationBridgeView（SwiftUI 容器，把 session 推回 bridge）

@available(macOS 15.0, *)
struct TranslationBridgeView: View {
    let bridge: TranslationBridge

    var body: some View {
        // generation 变化 → view identity 变化 → body 重执行 → .translationTask 闭包重启
        EmptyView()
            .id(bridge.generation)
            .translationTask(bridge.pendingConfiguration) { session in
                LogStore.shared.append(
                    "→ .translationTask 闭包启动（generation=\(bridge.generation)，config=\(String(describing: bridge.pendingConfiguration))）",
                    source: "SwiftUI"
                )
                bridge.deliverSession(session)
            }
    }
}

@available(macOS 15.0, *)
final class TranslationView: NSView {
    private let sourcePopup = NSPopUpButton()
    private let targetPopup = NSPopUpButton()
    private let inputScroll = NSScrollView()
    private let inputView = NSTextView()
    private let outputScroll = NSScrollView()
    private let outputView = NSTextView()
    private let statusLabel = NSTextField(labelWithString: "")
    private let translateButton = NSButton(title: "翻译", target: nil, action: nil)
    private let copyButton = NSButton(title: "复制结果", target: nil, action: nil)
    private let clearButton = NSButton(title: "清空", target: nil, action: nil)
    private let downloadButton = NSButton(title: "打开系统设置", target: nil, action: nil)

    // bridge: 长生命周期的 @MainActor actor-style 对象；每次翻译改它的 generation
    // hosting view: 每次翻译都销毁重建（旧 removeFromSuperview + 新 addSubview），
    //   强制 SwiftUI 重新执行 .translationTask 闭包（这是修复长文本卡死的关键）
    //
    // bridge: 每次翻译都**新建一个**，让旧的 TranslationBridge 实例彻底释放，
    //   避免 Translation framework 内部因同一 session 复用产生的资源竞争
    //   （实测发现 session 复用偶尔会导致 session.translate 卡死 30s）
    private var bridge: TranslationBridge?
    private var bridgeHostingView: NSHostingView<TranslationBridgeView>?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
        buildUI()
    }

    required init?(coder: NSCoder) { fatalError() }

    private func buildUI() {
        // 源/目标下拉
        for choice in sourceLanguages { sourcePopup.addItem(withTitle: choice.display) }
        // 默认源：English
        if let idx = sourceLanguages.firstIndex(where: { $0.identifier == "en" }) {
            sourcePopup.selectItem(at: idx)
        } else {
            sourcePopup.selectItem(at: 0)
        }
        sourcePopup.translatesAutoresizingMaskIntoConstraints = false

        for choice in targetLanguages { targetPopup.addItem(withTitle: choice.display) }
        // 默认目标：中文 (简体)
        if let idx = targetLanguages.firstIndex(where: { $0.identifier == "zh-Hans" }) {
            targetPopup.selectItem(at: idx)
        }
        targetPopup.translatesAutoresizingMaskIntoConstraints = false

        let sourceLabel = NSTextField(labelWithString: "源:")
        sourceLabel.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        sourceLabel.translatesAutoresizingMaskIntoConstraints = false

        let targetLabel = NSTextField(labelWithString: "目标:")
        targetLabel.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        targetLabel.translatesAutoresizingMaskIntoConstraints = false

        // Swap 按钮：源/目标互换，并把当前翻译结果填入输入框（如果有）
        let swapButton = NSButton(title: "⇄", target: self, action: #selector(swapClicked))
        swapButton.bezelStyle = .rounded
        swapButton.font = NSFont.systemFont(ofSize: 16, weight: .semibold)
        swapButton.toolTip = "互换源和目标语种"
        swapButton.translatesAutoresizingMaskIntoConstraints = false

        // 输入框
        configureScrollView(inputScroll, with: inputView, editable: true)
        inputView.font = NSFont.systemFont(ofSize: 13)
        inputView.string = "Hello, World!\n\nThe quick brown fox jumps over the lazy dog."

        // 输出框
        configureScrollView(outputScroll, with: outputView, editable: false)
        outputView.font = NSFont.systemFont(ofSize: 13)
        outputView.textColor = .secondaryLabelColor

        // 状态行
        statusLabel.font = NSFont.systemFont(ofSize: 11)
        statusLabel.textColor = .systemBlue
        statusLabel.lineBreakMode = .byTruncatingTail
        statusLabel.translatesAutoresizingMaskIntoConstraints = false
        statusLabel.stringValue = "就绪。点击「翻译」按钮开始。"

        // 按钮
        translateButton.target = self
        translateButton.action = #selector(translateClicked)
        translateButton.bezelStyle = .rounded
        translateButton.translatesAutoresizingMaskIntoConstraints = false

        copyButton.target = self
        copyButton.action = #selector(copyClicked)
        copyButton.bezelStyle = .rounded
        copyButton.translatesAutoresizingMaskIntoConstraints = false

        clearButton.target = self
        clearButton.action = #selector(clearClicked)
        clearButton.bezelStyle = .rounded
        clearButton.translatesAutoresizingMaskIntoConstraints = false

        downloadButton.target = self
        downloadButton.action = #selector(openSettingsClicked)
        downloadButton.bezelStyle = .rounded
        downloadButton.translatesAutoresizingMaskIntoConstraints = false
        downloadButton.isHidden = true

        addSubview(sourceLabel)
        addSubview(sourcePopup)
        addSubview(swapButton)
        addSubview(targetLabel)
        addSubview(targetPopup)
        addSubview(inputScroll)
        addSubview(translateButton)
        addSubview(copyButton)
        addSubview(clearButton)
        addSubview(downloadButton)
        addSubview(statusLabel)
        addSubview(outputScroll)
        installBridgeHostingView()

        NSLayoutConstraint.activate([
            sourceLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            sourceLabel.topAnchor.constraint(equalTo: topAnchor, constant: 16),

            sourcePopup.leadingAnchor.constraint(equalTo: sourceLabel.trailingAnchor, constant: 8),
            sourcePopup.centerYAnchor.constraint(equalTo: sourceLabel.centerYAnchor),
            sourcePopup.widthAnchor.constraint(equalToConstant: 160),

            swapButton.leadingAnchor.constraint(equalTo: sourcePopup.trailingAnchor, constant: 8),
            swapButton.centerYAnchor.constraint(equalTo: sourceLabel.centerYAnchor),
            swapButton.widthAnchor.constraint(equalToConstant: 36),

            targetLabel.leadingAnchor.constraint(equalTo: swapButton.trailingAnchor, constant: 8),
            targetLabel.centerYAnchor.constraint(equalTo: sourceLabel.centerYAnchor),

            targetPopup.leadingAnchor.constraint(equalTo: targetLabel.trailingAnchor, constant: 8),
            targetPopup.centerYAnchor.constraint(equalTo: sourceLabel.centerYAnchor),
            targetPopup.widthAnchor.constraint(equalToConstant: 180),

            inputScroll.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            inputScroll.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            inputScroll.topAnchor.constraint(equalTo: sourceLabel.bottomAnchor, constant: 12),
            inputScroll.heightAnchor.constraint(equalToConstant: 90),

            translateButton.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            translateButton.topAnchor.constraint(equalTo: inputScroll.bottomAnchor, constant: 12),

            copyButton.leadingAnchor.constraint(equalTo: translateButton.trailingAnchor, constant: 8),
            copyButton.centerYAnchor.constraint(equalTo: translateButton.centerYAnchor),

            clearButton.leadingAnchor.constraint(equalTo: copyButton.trailingAnchor, constant: 8),
            clearButton.centerYAnchor.constraint(equalTo: translateButton.centerYAnchor),

            downloadButton.leadingAnchor.constraint(equalTo: clearButton.trailingAnchor, constant: 8),
            downloadButton.centerYAnchor.constraint(equalTo: translateButton.centerYAnchor),

            statusLabel.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            statusLabel.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            statusLabel.topAnchor.constraint(equalTo: translateButton.bottomAnchor, constant: 12),

            outputScroll.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            outputScroll.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            outputScroll.topAnchor.constraint(equalTo: statusLabel.bottomAnchor, constant: 6),
            outputScroll.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -16),
        ])
    }

    private func configureScrollView(_ scroll: NSScrollView, with textView: NSTextView, editable: Bool) {
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        scroll.translatesAutoresizingMaskIntoConstraints = false

        textView.isEditable = editable
        textView.isSelectable = true
        textView.isRichText = false
        textView.allowsUndo = editable
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.frame = NSRect(x: 0, y: 0, width: 100, height: 100)

        scroll.documentView = textView
    }

    private func installBridgeHostingView() {
        // 关键：每次翻译都**销毁旧 hosting view**，重新 addSubview 一个新的。
        // NSHostingView.rootView = ... 赋值在某些情况下 SwiftUI 不会重建 view tree，
        // 导致 .translationTask 闭包不重启。销毁重建是 100% 触发的方式。
        if let old = bridgeHostingView {
            old.removeFromSuperview()
            bridgeHostingView = nil
        }

        // 每次新建 TranslationBridge，避免 session 复用导致 framework 内部卡死
        let newBridge = TranslationBridge()
        self.bridge = newBridge

        let hosting = NSHostingView(rootView: TranslationBridgeView(bridge: newBridge))
        hosting.frame = NSRect(x: 0, y: 0, width: 1, height: 1)
        hosting.translatesAutoresizingMaskIntoConstraints = false
        addSubview(hosting)
        bridgeHostingView = hosting
    }

    // MARK: 按钮动作

    @objc private func translateClicked() {
        let inputText = inputView.string
        LogStore.shared.append("按钮点击：text 长度=\(inputText.count) 字符", source: "AppKit")

        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            setStatus("请先输入要翻译的文本", color: .systemOrange)
            LogStore.shared.append("拒绝：空文本", source: "AppKit")
            return
        }

        let sourceIdx = sourcePopup.indexOfSelectedItem
        let targetIdx = targetPopup.indexOfSelectedItem
        guard sourceIdx >= 0, targetIdx >= 0,
              sourceIdx < sourceLanguages.count, targetIdx < targetLanguages.count else {
            setStatus("请选择源和目标语种", color: .systemOrange)
            return
        }

        let sourceChoice = sourceLanguages[sourceIdx]
        let targetChoice = targetLanguages[targetIdx]
        let startTime = Date()

        setStatus("正在准备翻译（macOS 15+ 通过 SwiftUI .translationTask 桥接）...", color: .systemBlue)
        translateButton.isEnabled = false
        downloadButton.isHidden = true

        LogStore.shared.append(
            "语种对：\(sourceChoice.identifier) → \(targetChoice.identifier)",
            source: "AppKit"
        )

        // 启动语种可用性预检查（在 AppKit 端做，避免无谓的 SwiftUI 触发）
        Task { @MainActor in
            let resolvedSource: Locale.Language = (sourceChoice.identifier == "auto")
                ? Locale.current.language
                : sourceChoice.language

            LogStore.shared.append("→ LanguageAvailability.status(...) 开始", source: "AppKit")
            let availability = LanguageAvailability()
            let status = await availability.status(from: resolvedSource, to: targetChoice.language)
            LogStore.shared.append("← LanguageAvailability 返回 status=\(status)", source: "AppKit")

            switch status {
            case .unsupported:
                self.translateButton.isEnabled = true
                self.setStatus("\(sourceChoice.display) → \(targetChoice.display) 语种对不支持", color: .systemRed)
                return
            case .supported:
                // .supported 可能是瞬时状态：macOS 系统层可能在后台主动下载翻译语种包。
                // 重试 5 次（每次 0.5s 间隔），给系统机会"准备就绪"。
                self.setStatus("首次检查显示未下载，重试中...", color: .systemBlue)
                LogStore.shared.append(
                    ".supported 状态，启用 5 次重试（间隔 0.5s）",
                    source: "AppKit"
                )
                var retryCount = 0
                var finalStatus: LanguageAvailability.Status = .supported
                while retryCount < 5 {
                    try? await Task.sleep(nanoseconds: 500_000_000)
                    retryCount += 1
                    let retryStatus = await availability.status(
                        from: resolvedSource,
                        to: targetChoice.language
                    )
                    LogStore.shared.append(
                        "  重试 \(retryCount)/5: status=\(retryStatus)",
                        source: "AppKit"
                    )
                    if retryStatus == .installed {
                        finalStatus = .installed
                        break
                    }
                    if retryStatus == .unsupported {
                        finalStatus = .unsupported
                        break
                    }
                }
                if finalStatus != .installed {
                    self.translateButton.isEnabled = true
                    if finalStatus == .unsupported {
                        self.setStatus("\(sourceChoice.display) → \(targetChoice.display) 语种对不支持", color: .systemRed)
                    } else {
                        self.setStatus("\(targetChoice.display) 语种包未下载。点击下面按钮去系统设置。", color: .systemOrange)
                        self.downloadButton.isHidden = false
                    }
                    return
                }
                // 重试成功，继续走翻译流程
                LogStore.shared.append("✓ 重试 \(retryCount) 次后状态变 installed", source: "AppKit")
                break
            case .installed:
                break
            @unknown default:
                self.translateButton.isEnabled = true
                self.setStatus("未知的语种支持状态", color: .systemRed)
                return
            }

            // 语种可用：销毁重建 hosting view（让 SwiftUI 重新执行 .translationTask 闭包），
            // 然后 await bridge.requestTranslate 等结果。
            self.setStatus("正在翻译...", color: .systemBlue)
            LogStore.shared.append(
                "→ installBridgeHostingView() 销毁重建（触发 SwiftUI .translationTask 闭包重启）",
                source: "AppKit"
            )
            self.installBridgeHostingView()

            do {
                let config = TranslationSession.Configuration(
                    source: resolvedSource,
                    target: targetChoice.language
                )
                guard let bridge = self.bridge else {
                    self.setStatus("内部错误：bridge 未创建", color: .systemRed)
                    self.translateButton.isEnabled = true
                    return
                }
                let result = try await bridge.requestTranslate(text: inputText, configuration: config)
                let elapsed = String(format: "%.2f", Date().timeIntervalSince(startTime))
                LogStore.shared.append("✓ 翻译成功（总耗时 \(elapsed)s，target 长度=\(result.count)）", source: "AppKit")
                self.outputView.string = result
                self.outputView.textColor = .labelColor
                self.setStatus("翻译完成（\(sourceChoice.display) → \(targetChoice.display)）", color: .systemGreen)
            } catch {
                let elapsed = String(format: "%.2f", Date().timeIntervalSince(startTime))
                LogStore.shared.append("✗ 翻译失败（耗时 \(elapsed)s）err=\(error)", source: "AppKit")
                self.handleBridgeError(String(describing: error))
            }
            self.translateButton.isEnabled = true
        }
    }

    private func handleBridgeError(_ errString: String) {
        // errString 是 String(describing:) 输出，形如
        // TranslationError.unsupportedSourceLanguage(reason: nil)
        // 简单按关键字匹配做友好提示
        if errString.contains("unsupportedSourceLanguage") {
            setStatus("源语种不支持", color: .systemRed)
        } else if errString.contains("unsupportedTargetLanguage") {
            setStatus("目标语种不支持", color: .systemRed)
        } else if errString.contains("unableToIdentifyLanguage") {
            setStatus("无法识别输入文本的语种", color: .systemRed)
        } else if errString.contains("nothingToTranslate") {
            setStatus("没有要翻译的内容", color: .systemRed)
        } else if errString.contains("notInstalled") {
            setStatus("语种包未安装，请到系统设置下载", color: .systemRed)
            downloadButton.isHidden = false
        } else {
            setStatus("翻译失败: \(errString)", color: .systemRed)
        }
    }

    @objc private func copyClicked() {
        let text = outputView.string
        guard !text.isEmpty else { return }
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(text, forType: .string)
        setStatus("已复制到剪贴板", color: .systemGreen)
    }

    @objc private func clearClicked() {
        inputView.string = ""
        outputView.string = ""
        setStatus("已清空", color: .secondaryLabelColor)
    }

    @objc private func openSettingsClicked() {
        // macOS 15+ 的系统设置 deep link，跳到翻译语种
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.language") {
            NSWorkspace.shared.open(url)
        }
        setStatus("请在系统设置 → 语言与地区 → 翻译语言 中下载所需语种包", color: .systemOrange)
    }

    @objc private func swapClicked() {
        let sourceIdx = sourcePopup.indexOfSelectedItem
        let targetIdx = targetPopup.indexOfSelectedItem
        guard sourceIdx >= 0, targetIdx >= 0,
              sourceIdx < sourceLanguages.count, targetIdx < targetLanguages.count else {
            return
        }

        let sourceChoice = sourceLanguages[sourceIdx]
        let targetChoice = targetLanguages[targetIdx]

        // 新源 = 当前目标的语种；新目标 = 当前源的语种。
        // 「自动」只在 sourceLanguages 里，targetLanguages 里没有 ——
        // 所以源当前若是「自动」，swap 后目标找不到「自动」就保留原目标并提示。
        let newSourceIdentifier = targetChoice.identifier
        let newTargetIdentifier = sourceChoice.identifier

        // 1. 找新源在 sourcePopup 里的 item index
        guard let newSourceItemIdx = (0..<sourcePopup.numberOfItems).first(where: {
            sourcePopup.itemTitle(at: $0) == sourceLanguages.first(where: { $0.identifier == newSourceIdentifier })?.display
        }) else {
            // 当前源是「自动」，目标列表里的 identifier 在源列表里找不到「自动」的对应 —— 这种情况下不能换
            if sourceChoice.identifier == "auto" {
                setStatus("「自动」无法换到目标位置（请先选具体语种）", color: .systemOrange)
                return
            }
            setStatus("无法找到目标语种的源选项", color: .systemOrange)
            return
        }

        // 2. 找新目标在 targetPopup 里的 item index
        guard let newTargetItemIdx = (0..<targetPopup.numberOfItems).first(where: {
            targetPopup.itemTitle(at: $0) == targetLanguages.first(where: { $0.identifier == newTargetIdentifier })?.display
        }) else {
            setStatus("「\(sourceChoice.display)」无法换到目标位置", color: .systemOrange)
            return
        }

        sourcePopup.selectItem(at: newSourceItemIdx)
        targetPopup.selectItem(at: newTargetItemIdx)

        // 同步翻译结果 → 输入框（如果有），方便用户重新翻译看到原文/译文对调
        if !outputView.string.isEmpty && outputView.textColor != .secondaryLabelColor {
            inputView.string = outputView.string
            outputView.string = ""
            outputView.textColor = .secondaryLabelColor
            setStatus("已互换语种，译文已填入输入框，点击「翻译」再译一次", color: .systemBlue)
        } else {
            setStatus("已互换源和目标语种", color: .systemBlue)
        }
    }

    // MARK: 翻译主流程（bridge 取代）

    private func setStatus(_ text: String, color: NSColor) {
        statusLabel.stringValue = text
        statusLabel.textColor = color
        LogStore.shared.append("status: \(text)", source: "Status")
    }

    // MARK: 日志（直接调 LogStore.shared.append 即可，不再需要 wrap）
}

// MARK: - LogStore（跨组件共享日志）

final class LogStore: @unchecked Sendable {
    static let shared = LogStore()
    private let lock = NSLock()
    private var lines: [String] = []
    var onChange: (() -> Void)?

    /// 全局开关：关掉时 append 静默忽略，不入队也不通知 UI
    var enabled: Bool = true

    func append(_ message: String, source: String = "App") {
        guard enabled else { return }
        lock.lock()
        let ts = Self.timestamp()
        let line = "[\(ts)] [\(source)] \(message)"
        lines.append(line)
        if lines.count > 500 { lines.removeFirst(lines.count - 500) }
        lock.unlock()
        let cb = onChange
        DispatchQueue.main.async { cb?() }
    }

    func snapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        return lines.joined(separator: "\n")
    }

    func clear() {
        lock.lock()
        lines.removeAll()
        lock.unlock()
        let cb = onChange
        DispatchQueue.main.async { cb?() }
    }

    private static func timestamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "HH:mm:ss.SSS"
        return f.string(from: Date())
    }
}

// MARK: - LogView（日志 Tab 的内容）

final class LogView: NSView {
    private let scroll = NSScrollView()
    private let textView = NSTextView()
    private let clearButton = NSButton(title: "清空日志", target: nil, action: nil)
    private let copyButton = NSButton(title: "复制全部", target: nil, action: nil)
    private var refreshTimer: Timer?

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        translatesAutoresizingMaskIntoConstraints = false
        buildUI()
        startRefreshing()
    }

    required init?(coder: NSCoder) { fatalError() }

    deinit { refreshTimer?.invalidate() }

    private func buildUI() {
        // 顶部说明
        let header = NSTextField(labelWithString: "日志（Translation 流程 + AppKit 桥接节点）")
        header.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        header.textColor = .secondaryLabelColor
        header.translatesAutoresizingMaskIntoConstraints = false

        // 按钮
        clearButton.target = self
        clearButton.action = #selector(clearClicked)
        clearButton.bezelStyle = .rounded
        clearButton.translatesAutoresizingMaskIntoConstraints = false

        copyButton.target = self
        copyButton.action = #selector(copyClicked)
        copyButton.bezelStyle = .rounded
        copyButton.translatesAutoresizingMaskIntoConstraints = false

        // 文本框
        scroll.hasVerticalScroller = true
        scroll.borderType = .bezelBorder
        scroll.translatesAutoresizingMaskIntoConstraints = false

        textView.isEditable = false
        textView.isSelectable = true
        textView.isRichText = false
        textView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        textView.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        textView.textContainer?.widthTracksTextView = true
        textView.frame = NSRect(x: 0, y: 0, width: 100, height: 100)
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(width: CGFloat.greatestFiniteMagnitude, height: CGFloat.greatestFiniteMagnitude)
        textView.autoresizingMask = [.width]
        textView.isVerticallyResizable = true
        textView.isHorizontallyResizable = false
        scroll.documentView = textView

        addSubview(header)
        addSubview(copyButton)
        addSubview(clearButton)
        addSubview(scroll)

        NSLayoutConstraint.activate([
            header.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            header.topAnchor.constraint(equalTo: topAnchor, constant: 16),

            copyButton.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            copyButton.centerYAnchor.constraint(equalTo: header.centerYAnchor),

            clearButton.trailingAnchor.constraint(equalTo: copyButton.leadingAnchor, constant: -8),
            clearButton.centerYAnchor.constraint(equalTo: header.centerYAnchor),

            scroll.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 16),
            scroll.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -16),
            scroll.topAnchor.constraint(equalTo: header.bottomAnchor, constant: 12),
            scroll.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -16),
        ])

        textView.string = LogStore.shared.snapshot()
    }

    @objc private func clearClicked() {
        LogStore.shared.clear()
        textView.string = ""
    }

    @objc private func copyClicked() {
        let pb = NSPasteboard.general
        pb.clearContents()
        pb.setString(textView.string, forType: .string)
        LogStore.shared.append("日志已复制到剪贴板（\(textView.string.count) 字符）", source: "Log")
    }

    private func startRefreshing() {
        // 启动时打一行
        LogStore.shared.append("=== LogView 启动，桥接 NSHostingView ===", source: "Log")

        refreshTimer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] _ in
            guard let self = self else { return }
            let snap = LogStore.shared.snapshot()
            if snap != self.textView.string {
                self.textView.string = snap
                self.textView.scrollToEndOfDocument(nil)
            }
        }
        if let t = refreshTimer {
            RunLoop.main.add(t, forMode: .common)
        }
    }
}

// MARK: - SettingsPanelController（偏好设置弹窗）

final class SettingsPanelController: NSObject, NSWindowDelegate {
    static let shared = SettingsPanelController()
    private var panel: NSPanel?

    func show(in parentWindow: NSWindow) {
        if let existing = panel {
            existing.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
            return
        }

        let contentRect = NSRect(x: 0, y: 0, width: 360, height: 140)
        let p = NSPanel(
            contentRect: contentRect,
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        p.title = "偏好设置"
        p.isFloatingPanel = true
        p.hidesOnDeactivate = false
        p.delegate = self

        let contentView = NSView(frame: contentRect)

        let header = NSTextField(labelWithString: "显示选项")
        header.font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        header.textColor = .labelColor
        header.translatesAutoresizingMaskIntoConstraints = false

        // 日志开关 checkbox
        let logCheckbox = NSButton(checkboxWithTitle: "启用日志写入", target: self, action: #selector(toggleLogWrite(_:)))
        logCheckbox.state = AppPreferences.enableLogWrite ? .on : .off
        logCheckbox.font = NSFont.systemFont(ofSize: 12)
        logCheckbox.translatesAutoresizingMaskIntoConstraints = false

        let hint = NSTextField(labelWithString: "关闭后 Translation 流程不再写入日志。日志 Tab 仍可查看关闭前的历史")
        hint.font = NSFont.systemFont(ofSize: 11)
        hint.textColor = .secondaryLabelColor
        hint.lineBreakMode = .byWordWrapping
        hint.maximumNumberOfLines = 2
        hint.translatesAutoresizingMaskIntoConstraints = false

        let closeButton = NSButton(title: "关闭", target: self, action: #selector(closeClicked))
        closeButton.bezelStyle = .rounded
        closeButton.keyEquivalent = "\u{1b}"  // Esc
        closeButton.translatesAutoresizingMaskIntoConstraints = false

        contentView.addSubview(header)
        contentView.addSubview(logCheckbox)
        contentView.addSubview(hint)
        contentView.addSubview(closeButton)

        NSLayoutConstraint.activate([
            header.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            header.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 18),

            logCheckbox.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            logCheckbox.topAnchor.constraint(equalTo: header.bottomAnchor, constant: 12),

            hint.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 36),
            hint.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            hint.topAnchor.constraint(equalTo: logCheckbox.bottomAnchor, constant: 6),

            closeButton.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            closeButton.bottomAnchor.constraint(equalTo: contentView.bottomAnchor, constant: -16),
        ])

        p.contentView = contentView

        // 居中到父窗口
        let parentFrame = parentWindow.frame
        let x = parentFrame.origin.x + (parentFrame.width - contentRect.width) / 2
        let y = parentFrame.origin.y + (parentFrame.height - contentRect.height) / 2
        p.setFrameOrigin(NSPoint(x: x, y: y))

        p.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        panel = p
    }

    @objc private func toggleLogWrite(_ sender: NSButton) {
        AppPreferences.enableLogWrite = (sender.state == .on)
        // 这条日志有可能因关掉开关而不会被记下 —— 这是预期行为，
        // 但 LogView 的 refresh timer 还会继续把现有快照刷上去
        LogStore.shared.append(
            "设置变更：enableLogWrite = \(AppPreferences.enableLogWrite)",
            source: "Settings"
        )
        NotificationCenter.default.post(name: AppPreferences.didChange, object: nil)
    }

    @objc private func closeClicked() {
        panel?.orderOut(nil)
    }

    func windowWillClose(_ notification: Notification) {
        panel = nil
    }
}

// MARK: - AppPreferences（应用级偏好）

enum AppPreferences {
    private static let key = "enableLogWrite"

    /// 是否往 LogStore 写日志（开关关掉时 LogStore.append 静默忽略）
    static var enableLogWrite: Bool {
        get {
            if UserDefaults.standard.object(forKey: key) == nil { return true }  // 默认开
            return UserDefaults.standard.bool(forKey: key)
        }
        set {
            UserDefaults.standard.set(newValue, forKey: key)
            LogStore.shared.enabled = newValue
        }
    }

    /// 启动时让 LogStore 反映当前偏好
    static func applyOnLaunch() {
        LogStore.shared.enabled = enableLogWrite
    }

    static let didChange = Notification.Name("AppPreferencesDidChange")
}

// MARK: - AppDelegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var helloView: NSView!
    var translationView: NSView!
    var logView: NSView!
    let segmented = NSSegmentedControl(labels: ["Hello", "翻译", "日志"], trackingMode: .selectOne, target: nil, action: nil)

    func applicationDidFinishLaunching(_ notification: Notification) {
        // 加载 Dock 图标
        if let iconURL = Bundle.main.url(forResource: "AppIcon", withExtension: "icns"),
           let iconImage = NSImage(contentsOf: iconURL) {
            NSApplication.shared.applicationIconImage = iconImage
        }

        // 让 LogStore 反映偏好（关掉开关 → 不写日志）
        AppPreferences.applyOnLaunch()

        let contentRect = NSRect(x: 0, y: 0, width: 720, height: 520)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Hello World + Translation"
        window.minSize = NSSize(width: 640, height: 480)
        window.center()

        // segmented control：固定 3 段
        segmented.selectedSegment = 0
        segmented.segmentStyle = .rounded
        segmented.translatesAutoresizingMaskIntoConstraints = false
        segmented.target = self
        segmented.action = #selector(tabChanged)

        // 三个子 view
        helloView = HelloView()
        if #available(macOS 15.0, *) {
            translationView = TranslationView(frame: .zero)
        } else {
            translationView = NSView()
        }
        translationView.isHidden = true

        logView = LogView(frame: .zero)
        logView.isHidden = true

        let separator = NSBox()
        separator.boxType = .separator
        separator.translatesAutoresizingMaskIntoConstraints = false

        let containerView = window.contentView!
        containerView.addSubview(segmented)
        containerView.addSubview(separator)
        containerView.addSubview(helloView)
        containerView.addSubview(translationView)
        containerView.addSubview(logView)

        NSLayoutConstraint.activate([
            segmented.leadingAnchor.constraint(equalTo: containerView.leadingAnchor, constant: 16),
            segmented.topAnchor.constraint(equalTo: containerView.topAnchor, constant: 12),

            separator.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            separator.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            separator.topAnchor.constraint(equalTo: segmented.bottomAnchor, constant: 10),

            helloView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            helloView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            helloView.topAnchor.constraint(equalTo: separator.bottomAnchor, constant: 4),
            helloView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor),

            translationView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            translationView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            translationView.topAnchor.constraint(equalTo: separator.bottomAnchor, constant: 4),
            translationView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor),

            logView.leadingAnchor.constraint(equalTo: containerView.leadingAnchor),
            logView.trailingAnchor.constraint(equalTo: containerView.trailingAnchor),
            logView.topAnchor.constraint(equalTo: separator.bottomAnchor, constant: 4),
            logView.bottomAnchor.constraint(equalTo: containerView.bottomAnchor),
        ])

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func tabChanged() {
        let idx = segmented.selectedSegment
        helloView.isHidden = (idx != 0)
        translationView.isHidden = (idx != 1)
        logView.isHidden = (idx != 2)
    }

    @objc func openSettings(_ sender: Any?) {
        SettingsPanelController.shared.show(in: window)
    }

    @objc func quitApp() {
        NSApp.terminate(nil)
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.regular)
installMainMenu()
app.run()

// MARK: - 主菜单

// 没菜单栏时，NSTextView 的 ⌘C/⌘V/⌘X/⌘A/⌘Z 不会触发。
// Cocoa 设计：菜单 item 的 keyEquivalent 是 first-responder chain 的入口。
// 即使菜单永远不用，菜单存在就能让文本框的标准编辑快捷键工作。
func installMainMenu() {
    let mainMenu = NSMenu()

    // App 菜单（带 Quit）
    let appMenuItem = NSMenuItem()
    let appMenu = NSMenu()
    let appName = ProcessInfo.processInfo.processName

    appMenu.addItem(
        withTitle: "关于 \(appName)",
        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
        keyEquivalent: ""
    )
    appMenu.addItem(NSMenuItem.separator())
    appMenu.addItem(
        withTitle: "隐藏 \(appName)",
        action: #selector(NSApplication.hide(_:)),
        keyEquivalent: "h"
    )
    appMenu.addItem(NSMenuItem.separator())
    appMenu.addItem(
        withTitle: "退出 \(appName)",
        action: #selector(NSApplication.terminate(_:)),
        keyEquivalent: "q"
    )
    appMenuItem.submenu = appMenu
    mainMenu.addItem(appMenuItem)

    // Edit 菜单（核心：让 ⌘C/⌘V 在 NSTextView 里工作）
    let editMenuItem = NSMenuItem()
    let editMenu = NSMenu(title: "编辑")

    editMenu.addItem(
        withTitle: "撤销",
        action: Selector(("undo:")),  // NSTextView 第一响应者
        keyEquivalent: "z"
    )
    editMenu.addItem(
        withTitle: "重做",
        action: Selector(("redo:")),
        keyEquivalent: "Z"  // ⌘⇧Z
    )
    editMenu.addItem(NSMenuItem.separator())
    editMenu.addItem(
        withTitle: "剪切",
        action: #selector(NSText.cut(_:)),
        keyEquivalent: "x"
    )
    editMenu.addItem(
        withTitle: "拷贝",
        action: #selector(NSText.copy(_:)),
        keyEquivalent: "c"
    )
    editMenu.addItem(
        withTitle: "粘贴",
        action: #selector(NSText.paste(_:)),
        keyEquivalent: "v"
    )
    editMenu.addItem(
        withTitle: "全选",
        action: #selector(NSText.selectAll(_:)),
        keyEquivalent: "a"
    )
    editMenuItem.submenu = editMenu
    mainMenu.addItem(editMenuItem)

    // 设置菜单（弹窗开关日志 Tab）
    let settingsMenuItem = NSMenuItem()
    let settingsMenu = NSMenu(title: "设置")
    settingsMenu.addItem(
        withTitle: "偏好设置…",
        action: #selector(AppDelegate.openSettings(_:)),
        keyEquivalent: ","
    )
    settingsMenuItem.submenu = settingsMenu
    mainMenu.addItem(settingsMenuItem)

    // Window 菜单（让 ⌘W 关窗、⌘N/N 行为符合 macOS 习惯）
    let windowMenuItem = NSMenuItem()
    let windowMenu = NSMenu(title: "窗口")
    windowMenu.addItem(
        withTitle: "最小化",
        action: #selector(NSWindow.performMiniaturize(_:)),
        keyEquivalent: "m"
    )
    windowMenu.addItem(
        withTitle: "全部置前",
        action: #selector(NSApplication.arrangeInFront(_:)),
        keyEquivalent: ""
    )
    windowMenuItem.submenu = windowMenu
    mainMenu.addItem(windowMenuItem)
    NSApp.windowsMenu = windowMenu

    NSApp.mainMenu = mainMenu
}
