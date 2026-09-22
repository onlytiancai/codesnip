import Cocoa

// 启动时读 SDK 与 CLT 路径，作为窗口副标题展示
// 这些值若取不到就显示 "unknown"，可作为 CLT 是否真在工作的二级信号
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

final class AppDelegate: NSObject, NSApplicationDelegate {
    var window: NSWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Bundle 化后才有 bundle ID 和 Resources 加载能力
        let bundleId = Bundle.main.bundleIdentifier ?? "(no bundle ID — CLI 进程)"
        let resourcesPath = Bundle.main.resourcePath ?? "(no resourcePath)"

        // 加载 Resources/AppIcon.icns 设为 Dock 图标
        if let iconURL = Bundle.main.url(forResource: "AppIcon", withExtension: "icns"),
           let iconImage = NSImage(contentsOf: iconURL) {
            NSApplication.shared.applicationIconImage = iconImage
        }

        let contentRect = NSRect(x: 0, y: 0, width: 520, height: 320)
        window = NSWindow(
            contentRect: contentRect,
            styleMask: [.titled, .closable, .miniaturizable],
            backing: .buffered,
            defer: false
        )
        window.title = "Hello World"
        window.center()

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

        // Bundle 化特有的展示信息
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

        let quitButton = NSButton(title: "退出", target: self, action: #selector(quitApp))
        quitButton.bezelStyle = .rounded
        quitButton.translatesAutoresizingMaskIntoConstraints = false

        let view = window.contentView!
        view.addSubview(titleLabel)
        view.addSubview(subtitleLabel)
        view.addSubview(sdkLabel)
        view.addSubview(pathLabel)
        view.addSubview(bundleLabel)
        view.addSubview(resourcesLabel)
        view.addSubview(quitButton)

        NSLayoutConstraint.activate([
            titleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            titleLabel.topAnchor.constraint(equalTo: view.topAnchor, constant: 50),

            subtitleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            subtitleLabel.topAnchor.constraint(equalTo: titleLabel.bottomAnchor, constant: 10),

            bundleLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            bundleLabel.topAnchor.constraint(equalTo: subtitleLabel.bottomAnchor, constant: 14),

            sdkLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            sdkLabel.topAnchor.constraint(equalTo: bundleLabel.bottomAnchor, constant: 10),

            pathLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            pathLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            pathLabel.topAnchor.constraint(equalTo: sdkLabel.bottomAnchor, constant: 4),

            resourcesLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 16),
            resourcesLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -16),
            resourcesLabel.topAnchor.constraint(equalTo: pathLabel.bottomAnchor, constant: 2),

            quitButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            quitButton.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -20),
        ])

        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
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
app.run()
