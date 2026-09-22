# macOS SDK 能力速览（Swift 学习版）

> 适用：macOS 27.0 SDK，本机 CLT 装了 332 个 framework。本文按"你能用这个 framework 做什么"分块，**不穷举**，只挑常用的和值得知道的。每一块都给你：能做什么、典型类、最小代码片段、对应 framework 名（编译时 `-framework XXX` 用）。

文档里所有 Swift 代码都用 `import XXX`，编译示例都假定你已经会写 `swiftc -framework XXX main.swift`。

---

## 0. 怎么读 framework 路径

```
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/
    Cocoa.framework
    SwiftUI.framework
    Metal.framework
    ...
```

- **Public framework**：可以直接 `import` 用
- **`_*` 前缀**（如 `_FoundationModels_AppKit.framework`）：私有/内部桥接，不要直接 import，会被拒
- **System framework**（如 `SystemConfiguration.framework`）：系统级别，往往配合命令行工具
- **Extension framework**：前缀+下划线表示专门给另一个 framework 扩展用的

查某个 framework 的所有 API：`find /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/XXX.framework -name "*.swiftinterface"` 看公开接口。

---

## 1. UI 层：窗口、控件、布局

### Cocoa / AppKit（最稳，老牌）

```swift
import Cocoa
let app = NSApplication.shared
app.setActivationPolicy(.regular)

let window = NSWindow(
    contentRect: NSRect(x: 0, y: 0, width: 400, height: 300),
    styleMask: [.titled, .closable, .miniaturizable, .resizable],
    backing: .buffered, defer: false)
window.title = "Hello"
window.center()
window.makeKeyAndOrderFront(nil)
app.run()
```

- 适用：需要精细控制原生 macOS 行为的应用（菜单栏、Dock 集成、自定义 NSView 绘制、拖拽、跨窗口通信）
- 强项：NSToolbar、NSMenu、NSOutlineView、NSTableView（带分组/排序）、Cocoa Bindings
- 弱项：声明式 UI 弱，写起来比 SwiftUI 啰嗦

### SwiftUI（现代，声明式）

```swift
import SwiftUI

struct ContentView: View {
    @State private var count = 0
    var body: some View {
        VStack {
            Text("Count: \(count)")
            Button("Click") { count += 1 }
        }
        .frame(width: 300, height: 200)
    }
}

NSApplication.shared.run {
    WindowGroup("Hello") { ContentView() }
}
```

- 适用：新项目、需要快速开发、跨 Apple 平台（iOS/macOS/tvOS/watchOS/visionOS 共用 UI 代码）
- 强项：实时预览、Modifier 组合、动画简洁、Combine 集成
- 弱项：CLI `swiftc` 编译 SwiftUI 麻烦（需要 `-parse-as-library` 等 trick），通常要 `.app` 包或 Package.swift

### 选哪个

- **学习/玩具**：先学 AppKit，理解 macOS 的事件循环、NSResponder 链、Auto Layout
- **新项目**：SwiftUI（除非要做 macOS 专属的复杂 UI，比如 Finder 风格的 OutlineView）

---

## 2. 系统能力（OS 级服务）

### Foundation

所有 Swift 程序都默认 import 的核心库。

```swift
import Foundation

// 文件
let url = URL(fileURLWithPath: "/tmp/test.txt")
try "hello".write(to: url, atomically: true, encoding: .utf8)

// 进程
let p = Process()
p.executableURL = URL(fileURLWithPath: "/bin/echo")
p.arguments = ["world"]
try p.run()
p.waitUntilExit()

// 日期
let now = Date()
let formatter = ISO8601DateFormatter()
print(formatter.string(from: now))

// JSON
struct User: Codable { let name: String; let age: Int }
let data = try JSONEncoder().encode(User(name: "Alice", age: 30))
```

包含：URL、Data、String、Date、Calendar、Timer、FileManager、URLSession（HTTP）、Process（子进程）、Pipe（管道）、NotificationCenter、UserDefaults、JSONEncoder/Decoder。

### CoreFoundation

C 层的 API，Swift 里也能用（带 `CF` 前缀或桥接类型）。

```swift
import CoreFoundation

// CFString → Swift String
let cfStr: CFString = "hello" as CFString
let swiftStr = cfStr as String

// 文件监听
let stream = CFReadStreamCreateWithFileURL(nil, URL(fileURLWithPath: "/tmp/test.txt") as CFURL)!
CFReadStreamOpen(stream)
// 配合 RunLoop 监听变化
```

适用：底层网络（CFNetwork）、文件流、定时器、字符串比较本地化（CFStringTokenizer 分词）、内存管理 C API。

---

## 3. 图形与渲染

### Core Graphics（Cocoa 里通过 NSGraphicsContext 暴露）

```swift
import CoreGraphics

let context = CGContext(
    data: nil, width: 200, height: 200,
    bitsPerComponent: 8, bytesPerRow: 0,
    space: CGColorSpaceCreateDeviceRGB(),
    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!

context.setFillColor(CGColor(red: 1, green: 0.5, blue: 0, alpha: 1))
context.fill(CGRect(x: 50, y: 50, width: 100, height: 100))

let image = context.makeImage()!
```

- 适用：图片处理、PDF 渲染、CGPath 矢量绘制、坐标系变换
- CPU 端渲染，不是 GPU

### Metal（GPU 加速）

```swift
import Metal

let device = MTLCreateSystemDefaultDevice()!  // Apple Silicon 上 GPU 设备
let queue = device.makeCommandQueue()!
```

- 适用：游戏、视频处理、计算着色器（GPU 并行计算）、3D
- 极底层，要写 shader（`.metal` 文件用 Metal Shading Language）

### MetalKit / SceneKit / SpriteKit

- **MetalKit**：方便用 Metal，显示 MTKView
- **SceneKit**：3D 场景图，加载 .scn 文件，做 3D 可视化
- **SpriteKit**：2D 游戏，物理引擎集成

### Core Image

GPU/CPU 滤镜管道，300+ 内置滤镜。

```swift
import CoreImage

let image = CIImage(image: NSImage(named: "photo.jpg"))!
let filter = CIFilter(name: "CISepiaTone")!
filter.setValue(image, forKey: kCIInputImageKey)
filter.setValue(0.8, forKey: kCIInputIntensityKey)
let output = filter.outputImage!
```

适用：图片批量处理、实时相机滤镜、照片 App 类。

---

## 4. 多媒体

### AVFoundation（音频/视频）

```swift
import AVFoundation

// 播放音频
let player = AVAudioPlayer(contentsOf: URL(fileURLWithPath: "/System/Library/Sounds/Glass.aiff"))!
player.play()

// 播放视频
let avPlayer = AVPlayer(url: URL(string: "https://example.com/video.mp4")!)
avPlayer.play()

// 录音
let recorder = try AVAudioRecorder(
    url: URL(fileURLWithPath: "/tmp/rec.m4a"),
    settings: [:])
recorder.record()
```

### AVKit

AVFoundation 的 UI 包装（AVPlayerView、AVCaptureView）。

### CoreAudio

C 层 API，音频流、MIDI、低延迟音频单元（AudioUnit）。

### CoreMedia / VideoToolbox

视频编解码（解码 H.264/HEVC、硬件加速 VideoToolbox）。

---

## 5. 网络

### URLSession（Foundation）

```swift
let task = URLSession.shared.dataTask(with: URL(string: "https://api.example.com")!) { data, response, error in
    if let data = data {
        print(String(data: data, encoding: .utf8)!)
    }
}
task.resume()
```

99% 的场景用这个就够了：HTTP/HTTPS、上传、下载、断点续传、后台下载。

### Network.framework（低层）

替代 BSD sockets 的 Swift 原生 API，支持连接池、TLS 配置、QUIC。

```swift
import Network
let connection = NWConnection(
    host: "example.com", port: 443,
    using: .tls)
connection.start(queue: .global())
```

### CFNetwork

C 层，更老但更强（HTTP cookie、认证挑战、低层 socket）。

### Bonjour（NSNetService / NetService）

本地网络服务发现（mDNS），比如局域网打印机、文件共享。

### WebSocket

URLSession 的 `.webSocketTask` 或第三方 Starscream。

---

## 6. 硬件与外设

### AVFoundation（摄像头/麦克风）

```swift
import AVFoundation
let session = AVCaptureSession()
session.sessionPreset = .high
let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)!
session.addInput(try! AVCaptureDeviceInput(device: camera))
```

### CoreBluetooth

蓝牙低功耗（BLE）通信，做 IoT 工具、心率带、外设控制。

### IOKit

驱动级硬件访问（USB、PCIe、电源管理）。C 层，复杂。

### GameController

手柄、键盘、鼠标自定义按键映射。

### HID（Human Interface Device）

低层 USB HID 设备通信。

---

## 7. 文本与国际化

### Foundation 的 String + NSRegularExpression

```swift
import Foundation
let regex = try! NSRegularExpression(pattern: #"(\d+)-(\d+)"#)
let matches = regex.matches(in: "abc 123-456", range: NSRange(location: 0, length: 13))
```

### NaturalLanguage

```swift
import NaturalLanguage

let tagger = NLTagger(tagSchemes: [.language, .sentimentScore, .nameType])
tagger.string = "Apple Inc. is headquartered in Cupertino."
tagger.enumerateTags(in: tagger.string!.startIndex..<tagger.string!.endIndex,
                     unit: .word, scheme: .nameType) { tag, range in
    print("\(tagger.string![range]) = \(tag?.rawValue ?? "?")")
    return true
}
```

- 语言识别、词性标注、命名实体识别（人名/地名/机构名）、情感分析
- 离线，本地 ML 模型

### CoreNLP（私有）

Apple 内部 NLP 框架，不暴露 public API。

---

## 8. 机器学习

### Core ML

```swift
import CoreML
let model = try! MyClassifier(configuration: MLModelConfiguration())
let prediction = try! model.prediction(input: MyClassifierInput(...))
```

- 用 `.mlmodel` 文件（Create ML 训练或下载的 ONNX 转）
- 图像分类、文本分类、推荐模型
- Apple Silicon 上 GPU 加速

### Create ML

Swift DSL，**训练** Core ML 模型。Xcode UI 工具或在命令行跑 `xcrun createml`。

### Vision

```swift
import Vision

let request = VNRecognizeTextRequest { request, error in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for obs in observations {
        print(obs.topCandidates(1).first?.string ?? "")
    }
}

let handler = VNImageRequestHandler(url: URL(fileURLWithPath: "/tmp/photo.jpg"))
try? handler.perform([request])
```

- OCR、人脸检测、人脸识别、矩形检测、条形码、姿势估计、图像相似度
- 离线，Apple Neural Engine 加速

### SoundAnalysis

音频分类（环境声音、警报声、音乐分类）。

### Translation

OS 26+ 内置实时翻译框架。

---

## 9. 系统集成

### UserNotifications

```swift
import UserNotifications
let center = UNUserNotificationCenter.current()
center.requestAuthorization(options: [.alert, .sound]) { _, _ in }
center.add(UNNotificationRequest(
    identifier: UUID().uuidString,
    content: UNMutableNotificationContent().apply {
        $0.title = "Hello"
        $0.body = "World"
    },
    trigger: nil))
```

需要 bundle ID 才能稳定工作（UserNotifications 在 CLI 进程里也能弹，但需要 entitlements）。

### ServiceManagement

注册开机启动项（LaunchAgent / LaunchDaemon）。

```swift
import ServiceManagement
SMAppService.mainApp.register()
```

### AppKit 的菜单栏 / 状态栏项

```swift
let item = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
item.button?.title = "🌟"
item.menu = NSMenu()
```

### Spotlight

通过 CoreSpotlight 把 App 内容索引进系统搜索。

### Quick Look

```swift
import QuickLook
QLPreviewController.shared.reloadData()  // 缩略图预览
```

### WidgetKit

桌面小组件，需要 app extension（不能用 CLI 进程）。

---

## 10. 数据持久化

### Core Data

```swift
import CoreData

let container = NSPersistentContainer(name: "Model")
container.loadPersistentStores { _, error in ... }

let ctx = container.viewContext
let entity = NSEntityDescription.entity(forEntityName: "User", in: ctx)!
let user = NSManagedObject(entity: entity, insertInto: ctx)
user.setValue("Alice", forKey: "name")
try ctx.save()
```

- 类似 ORM，SQLite 底层
- 适合结构化数据、需要查询/关联/版本迁移

### SwiftData（macOS 14+）

SwiftUI 原生，Core Data 的现代化封装：

```swift
@Model
class User { var name: String; var age: Int; init(name: String, age: Int) { ... } }
```

### FileManager / UserDefaults

`UserDefaults.standard.set("dark", forKey: "theme")` 存小配置，`FileManager` 管大文件。

### Keychain Services（Security framework）

存密码、token、证书。Sandbox 应用必备。

```swift
import Security
SecItemAdd([...] as CFDictionary, nil)
```

### CloudKit

云同步，跨设备共享数据。要 bundle ID 和 iCloud entitlements。

---

## 11. 安全与加密

### CryptoKit

```swift
import CryptoKit

// SHA-256
let hash = SHA256.hash(data: Data("hello".utf8))
print(hash.compactMap { String(format: "%02x", $0) }.joined())

// AES 加密
let key = SymmetricKey(size: .bits256)
let sealed = try AES.GCM.seal(Data("secret".utf8), using: key)

// 签名 (Curve25519, Ed25519)
let signingKey = Curve25519.Signing.PrivateKey()
let signature = try signingKey.signature(for: data)
```

- 哈希、对称加密（AES-GCM、ChaChaPoly）、非对称（Curve25519、P256）
- 替代老旧的 CommonCrypto

### LocalAuthentication

```swift
import LocalAuthentication
let ctx = LAContext()
ctx.evaluatePolicy(.deviceOwnerAuthentication, localizedReason: "解锁") { success, error in
    print(success ? "通过" : "失败: \(error)")
}
```

Touch ID / Face ID / 密码验证。

### Security.framework

证书、Keychain、Trust 评估、代码签名。

---

## 12. 并发与性能

### Swift Concurrency（async/await, Task）

```swift
func fetchData() async throws -> Data {
    let (data, _) = try await URLSession.shared.data(from: url)
    return data
}

Task {
    do { let data = try await fetchData() } catch { print(error) }
}
```

- Actor 模型防止数据竞争
- TaskGroup 并行多个任务

### Dispatch (GCD)

```swift
DispatchQueue.global().async {
    // 后台线程
    DispatchQueue.main.async {
        // 回到主线程更新 UI
    }
}
```

- 老牌，老代码里到处都是
- 新代码优先用 async/await

### Combine

```swift
import Combine
let publisher = URLSession.shared.dataTaskPublisher(for: url)
let cancellable = publisher
    .map(\.data)
    .sink { completion in ... } receiveValue: { data in ... }
```

- 响应式，SwiftUI 里特别好用

### Accelerate

- SIMD 数学库（vDSP）、线性代数（BLAS/LAPACK）、信号处理、图像卷积
- CPU 优化版，比手写循环快几个数量级

### Metal Performance Shaders

GPU 上的卷积、矩阵乘法，做机器学习推理加速。

---

## 13. 进程间通信

### DistributedNotificationCenter

跨进程通知（同机器不同 App 间广播）。

### XPC

```swift
import XPC
let conn = NSXPCConnection(serviceName: "com.example.helper")
conn.remoteObjectInterface = NSXPCInterface(with: HelperProtocol.self)
conn.resume()
```

- 沙盒应用的安全进程间通信
- 比 socket 简单，比直接调用安全

### Apple Events / AppleScript

```swift
NSWorkspace.shared.open(URL(string: "https://apple.com")!)
// 或 Automation.framework 发 AppleScript
tell application "Safari" to open location "https://..."
```

- 让你的 App 被 AppleScript 控制
- 或你的 App 控制其他 App

### Pasteboard（剪贴板）

```swift
import AppKit
let pb = NSPasteboard.general
pb.clearContents()
pb.setString("hello", forType: .string)
```

---

## 14. 调试与开发辅助

### os_log / OSLog

```swift
import os
let logger = Logger(subsystem: "com.example.app", category: "network")
logger.info("Request started")
logger.error("Failed: \(error.localizedDescription)")
```

- 替代 print，可分级（debug/info/error/fault），控制台.app 能按 subsystem/category 过滤
- 性能：debug 级会被 release build 剥掉

### MetricKit

应用性能指标（启动时间、卡顿、内存峰值、电池消耗）上报。

### Instruments

Xcode 自带，但命令行也能跑 `xcrun xctest` / `xcrun instruments`。

### ActivityKit

Live Activities（iOS 上的实时小组件，macOS 部分支持）。

---

## 15. 命令行/工具类 framework

### SystemConfiguration

读网络状态、代理设置、电源信息、主机名。

### IOKit

USB 设备列举、电源管理、磁盘 SMART。命令行工具常用。

### DiskArbitration

磁盘挂载/卸载事件监听。

### libxml2

XML 解析（命令行 `xmllint` 用的就是这个）。

### ScriptingBridge

控制其他 AppleScriptable App：

```swift
import ScriptingBridge
let safari = SBApplication(bundleIdentifier: "com.apple.Safari")!
safari.setValue(true, forKey: "running")
```

---

## 16. Apple 生态专属

### StoreKit

App 内购买、订阅验证。需要 bundle ID。

### PassKit

Apple Wallet / 票据。

### MusicKit / MediaPlayer

Apple Music API、播放库访问。

### MapKit

地图视图、地理编码、路径规划。

### Contacts / AddressBook

通讯录访问（AddressBook 旧，Contacts 新）。

### EventKit

日历、提醒事项。

### Photos / PhotosUI

照片库访问。

### HomeKit

智能家居控制。

### HealthKit

健康数据（macOS 14+ 才有，且要 entitlement）。

### GameKit

Game Center 排行榜、成就、多人对战。

---

## 17. SwiftUI 专属（仅 SwiftUI 项目用）

| Framework | 干嘛的 |
|---|---|
| SwiftUI | 声明式 UI |
| WidgetKit | 桌面小组件（要 App Extension） |
| StoreKit | App 内购 UI |
| PDFKit | PDF 查看器 |
| QuickLook | 预览 |
| Charts | 图表（iOS 16+ / macOS 13+） |
| MapKit | 地图 |
| RealityKit | 3D AR |
| SceneKit | 3D 场景 |
| SpriteKit | 2D 游戏 |

---

## 18. macOS 独有的"系统级"能力

其他平台没有，macOS 才有：

- **App Nap**：应用闲置时降低 CPU 优先级，系统会通知
- **Sandbox**：每个 App 沙盒隔离访问
- **Notarization**：Apple 公证后才能分发
- **Code Signing**：代码签名 + 团队身份
- **Hardened Runtime**：运行时完整性检查（防止动态链接注入）
- **Gatekeeper**：首次启动时验证开发者身份
- **XPC Services**：细粒度进程隔离
- **Login Items**：开机自启动管理
- **Power Assertions**：阻止系统睡眠
- **Accessibility API**：辅助功能（VoiceOver 朗读、其他 App 也能读你的 UI）
- **macOS App Extensions**：Finder 扩展、分享面板扩展、Today 小组件
- **Spotlight Importer**：自定义文件类型索引
- **Quick Look Generator**：自定义文件预览
- **Launch Services**：文件 ↔ App 关联
- **Bonjour**：本地服务发现
- **iCloud**：系统级云存储
- **Time Machine**：系统级备份（App 可以标记自己的文件让 TM 排除）
- **Auto Layout / SwiftUI**：macOS 上是最早支持的（iOS 是 2014 才有 Auto Layout）
- **Touch Bar**（仅 Intel/部分机型）：触控栏
- **Stage Manager / Spaces**：桌面分组

---

## 19. 怎么选 framework：实用决策树

```
想要做的事
├─ 画界面 ───────────── AppKit (成熟) / SwiftUI (新)
├─ 网络请求 ─────────── URLSession (99% 够用)
├─ 播放音视频 ───────── AVFoundation / AVKit
├─ 处理图片 ─────────── Core Image / Vision
├─ OCR / 人脸检测 ───── Vision
├─ ML 推理 ──────────── Core ML
├─ 加密哈希签名 ──────── CryptoKit
├─ 蓝牙 ─────────────── CoreBluetooth
├─ 摄像头/麦克风 ─────── AVFoundation
├─ 触控板/手势 ───────── AppKit (NSGestureRecognizer)
├─ 菜单栏图标 ────────── AppKit (NSStatusBar)
├─ 文件 I/O ─────────── Foundation (FileManager / Data)
├─ 多线程 ───────────── Swift Concurrency (新) / GCD (老)
├─ 数据持久化 ───────── Core Data / SwiftData / SQLite
├─ 跨进程通信 ────────── XPC (推荐) / Apple Events / Socket
├─ 开机启动 ──────────── ServiceManagement
├─ 系统通知 ─────────── UserNotifications
├─ 硬件控制 ──────────── IOKit (低层) / 各类专用 framework
└─ 命令行工具 ────────── Foundation + Swift Argument Parser (包)
```

---

## 20. 学习路径建议

1. **Foundation**（先精通）：String、URL、Date、JSON、Process、FileManager
2. **AppKit 或 SwiftUI**（二选一）：画第一个窗口
3. **URLSession**：发个 HTTP 请求
4. **Core Data / SwiftData**：存个文件
5. **Core Graphics + Core Animation**：画个自定义控件
6. **AVFoundation / Vision**：处理媒体
7. **Core ML**：跑个 ML 模型
8. **XPC + App Group**：做正经的多进程 App

---

## 21. 编译时怎么指定 framework

```bash
# 单 framework
swiftc -framework Foundation main.swift

# 多 framework
swiftc -framework Cocoa -framework AVFoundation -framework Vision main.swift

# 简写（自动从 import 推断）
swiftc main.swift   # 如果只用了 Foundation/Cocoa 经常能省 -framework
```

**坑**：某些 framework 必须显式写，编译器不会自动找到（特别是带 dlopen 路径的，比如 `MetalKit`）。出错提示 `umbrella header not found` 时就是忘了 `-framework`。

---

## 参考

- [Apple Developer Documentation](https://developer.apple.com/documentation/) 官方文档首页
- 本机路径：`file:///Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/`
- [Swift Standard Library](https://developer.apple.com/documentation/swift)（不是 framework，但所有代码都用）
- [Hacking with Swift](https://www.hackingwithswift.com/) 实战教程
- [Swift by Sundell](https://www.swiftbysundell.com/) 进阶文章

---

*按"能力"分块而不是按"framework 名字母"排序，是为了让你看到想做一件事的时候能直接找到对应那一节。*
