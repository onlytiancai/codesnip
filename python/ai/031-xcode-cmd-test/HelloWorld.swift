import Cocoa
import Translation

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

    // MARK: 按钮动作

    @objc private func translateClicked() {
        let inputText = inputView.string
        guard !inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            setStatus("请先输入要翻译的文本", color: .systemOrange)
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

        setStatus("正在检查语种支持...", color: .systemBlue)
        translateButton.isEnabled = false
        downloadButton.isHidden = true

        Task { @MainActor in
            await self.runTranslation(
                source: sourceChoice,
                target: targetChoice,
                text: inputText
            )
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

    // MARK: 翻译主流程

    private func runTranslation(source: LanguageChoice, target: LanguageChoice, text: String) async {
        let availability = LanguageAvailability()

        // 1. 决定源语种
        let resolvedSource: Locale.Language = (source.identifier == "auto")
            ? Locale.current.language
            : source.language

        // 2. 检查支持
        let status = await availability.status(from: resolvedSource, to: target.language)
        switch status {
        case .unsupported:
            translateButton.isEnabled = true
            setStatus("\(source.display) → \(target.display) 语种对不支持", color: .systemRed)
            return
        case .supported:
            translateButton.isEnabled = true
            setStatus("\(target.display) 语种包未下载。点击下面按钮去系统设置。", color: .systemOrange)
            downloadButton.isHidden = false
            return
        case .installed:
            break
        @unknown default:
            translateButton.isEnabled = true
            setStatus("未知的语种支持状态", color: .systemRed)
            return
        }

        // 3. 创建 session + 翻译
        setStatus("正在翻译...", color: .systemBlue)
        do {
            let session = TranslationSession(
                installedSource: resolvedSource,
                target: target.language
            )
            let response = try await session.translate(text)
            outputView.string = response.targetText
            outputView.textColor = .labelColor
            setStatus("翻译完成（\(response.sourceLanguage.maximalIdentifier) → \(response.targetLanguage.maximalIdentifier)）", color: .systemGreen)
        } catch let error as TranslationError {
            let msg: String
            switch error {
            case .unsupportedSourceLanguage:
                msg = "源语种不支持"
            case .unsupportedTargetLanguage:
                msg = "目标语种不支持"
            case .unableToIdentifyLanguage:
                msg = "无法识别输入文本的语种"
            case .nothingToTranslate:
                msg = "没有要翻译的内容"
            default:
                // 用 ~= pattern matching 检查 .notInstalled（避免 == 编译错误）
                if TranslationError.notInstalled ~= error {
                    msg = "语种包未安装，请到系统设置下载"
                    downloadButton.isHidden = false
                } else {
                    msg = "翻译失败: \(error.localizedDescription)"
                }
            }
            setStatus(msg, color: .systemRed)
        } catch {
            setStatus("翻译失败: \(error.localizedDescription)", color: .systemRed)
        }
        translateButton.isEnabled = true
    }

    private func setStatus(_ text: String, color: NSColor) {
        statusLabel.stringValue = text
        statusLabel.textColor = color
    }
}

// MARK: - AppDelegate

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!
    var helloView: NSView!
    var translationView: NSView!
    let segmented = NSSegmentedControl(labels: ["Hello", "翻译"], trackingMode: .selectOne, target: nil, action: nil)

    func applicationDidFinishLaunching(_ notification: Notification) {
        // 加载 Dock 图标
        if let iconURL = Bundle.main.url(forResource: "AppIcon", withExtension: "icns"),
           let iconImage = NSImage(contentsOf: iconURL) {
            NSApplication.shared.applicationIconImage = iconImage
        }

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

        // segmented control
        segmented.selectedSegment = 0
        segmented.segmentStyle = .rounded
        segmented.translatesAutoresizingMaskIntoConstraints = false
        segmented.target = self
        segmented.action = #selector(tabChanged)

        // 两个子 view
        helloView = HelloView()
        if #available(macOS 15.0, *) {
            translationView = TranslationView(frame: .zero)
        } else {
            // 理论上不会到这里，因为 LSMinimumSystemVersion = 15.0
            translationView = NSView()
        }
        translationView.isHidden = true

        let separator = NSBox()
        separator.boxType = .separator
        separator.translatesAutoresizingMaskIntoConstraints = false

        let containerView = window.contentView!
        containerView.addSubview(segmented)
        containerView.addSubview(separator)
        containerView.addSubview(helloView)
        containerView.addSubview(translationView)

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
        ])

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    @objc func tabChanged() {
        let idx = segmented.selectedSegment
        helloView.isHidden = (idx != 0)
        translationView.isHidden = (idx != 1)
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
