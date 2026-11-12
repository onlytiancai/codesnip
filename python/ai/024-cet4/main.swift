// VocabViewer — macOS 单词查看器（AppKit + 单文件 swiftc）
//
// 数据：复用 vocab_all.json（JSONL）+ vocab_all.idx（wordHead → 字节偏移）。
//      与 scripts/cet4.py 共用同一份索引文件。
//
// 编译：swiftc -sdk "$(xcrun --sdk macosx --show-sdk-path)" -framework AppKit \
//        main.swift -o VocabViewer
//
// 运行：./VocabViewer                          # 找 Bundle 同目录或 cwd 的 vocab_all.idx
//      ./VocabViewer --data /path/to/dir      # 自定义数据目录
//
// 关键陷阱：NSTextView 的 ⌘C/⌘V/⌘A/⌘Z 完全依赖主菜单里真有 Edit 子菜单，
// 手写 NSApplication 时必须显式装上。

import Foundation
import AppKit


// MARK: - Index ──────────────────────────────────────────────────────────────

/// 加载 vocab_all.idx，提供 wordHead → 字节偏移 映射与单行 seek 读取。
struct Index {
    let jsonURL: URL
    let heads: [String]
    let offsets: [String: Int]

    /// 在候选目录里找 vocab_all.idx 并加载。
    /// - Returns: nil 表示所有候选目录都没找到。
    static func load(jsonDir: URL) throws -> Index {
        let idxURL = jsonDir.appendingPathComponent("vocab_all.idx")
        let jsonURL = jsonDir.appendingPathComponent("vocab_all.json")

        guard FileManager.default.fileExists(atPath: idxURL.path) else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "找不到 \(idxURL.path)"]
            )
        }
        guard FileManager.default.fileExists(atPath: jsonURL.path) else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 2,
                userInfo: [NSLocalizedDescriptionKey: "找不到 \(jsonURL.path)"]
            )
        }

        let data = try Data(contentsOf: idxURL)
        guard let obj = try? JSONSerialization.jsonObject(with: data),
              let dict = obj as? [String: Any] else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 3,
                userInfo: [NSLocalizedDescriptionKey: "vocab_all.idx 不是合法 JSON"]
            )
        }
        guard (dict["version"] as? Int) == 1 else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 4,
                userInfo: [NSLocalizedDescriptionKey: "vocab_all.idx 版本不匹配（期望 1）"]
            )
        }
        guard let heads = dict["heads"] as? [String],
              let offsets = dict["offsets"] as? [String: Int] else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 5,
                userInfo: [NSLocalizedDescriptionKey: "vocab_all.idx 缺少 heads/offsets"
                ]
            )
        }
        guard heads.count == offsets.count else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 6,
                userInfo: [NSLocalizedDescriptionKey:
                    "vocab_all.idx 不一致：heads=\(heads.count), offsets=\(offsets.count)"
                ]
            )
        }
        return Index(jsonURL: jsonURL, heads: heads, offsets: offsets)
    }

    /// 按偏移读出对应行（JSON 顶层对象）。
    /// 与 scripts/cet4.py fetch_row 行为等价。
    func fetch(_ offset: Int) throws -> [String: Any] {
        let h = try FileHandle(forReadingFrom: jsonURL)
        defer { try? h.close() }
        try h.seek(toOffset: UInt64(offset))
        let rest = h.readDataToEndOfFile()
        // 取首个 \n 之前的内容；末行无 \n 时 rest 整体就是目标行
        if let nlIdx = rest.firstIndex(of: 0x0A) {
            let line = rest.subdata(in: 0..<nlIdx)
            return try parseLine(line)
        } else {
            return try parseLine(rest)
        }
    }

    private func parseLine(_ data: Data) throws -> [String: Any] {
        let obj = try JSONSerialization.jsonObject(with: data)
        guard let dict = obj as? [String: Any] else {
            throw NSError(
                domain: "VocabViewer.Index",
                code: 7,
                userInfo: [NSLocalizedDescriptionKey: "JSON 行不是对象"]
            )
        }
        return dict
    }

    /// 随机抽一个 wordHead。
    func randomHead() -> String {
        return heads.randomElement() ?? "a"
    }
}


// MARK: - Suggest ────────────────────────────────────────────────────────────

/// 找不到时的候选建议：与 scripts/cet4.py suggest_similar 行为等价。
enum Suggest {
    /// 长度倒序，匹配时取第一个且要求剥后剩余长度 ≥ 2
    static let SUFFIXES: [String] = [
        "ies", "ied", "ying", "ing", "ed", "er", "est", "es", "ly", "d", "s"
    ]

    /// 经典 DP + 两行滚动数组。词长 ≤ 30，无需更花哨的算法。
    static func levenshtein(_ a: String, _ b: String) -> Int {
        let aChars = Array(a)
        let bChars = Array(b)
        if aChars.isEmpty { return bChars.count }
        if bChars.isEmpty { return aChars.count }

        var prev = Array(0...bChars.count)
        var curr = Array(repeating: 0, count: bChars.count + 1)

        for i in 1...aChars.count {
            curr[0] = i
            for j in 1...bChars.count {
                let cost = aChars[i - 1] == bChars[j - 1] ? 0 : 1
                curr[j] = min(
                    curr[j - 1] + 1,         // insert
                    prev[j] + 1,             // delete
                    prev[j - 1] + cost       // substitute
                )
            }
            swap(&prev, &curr)
        }
        return prev[bChars.count]
    }

    /// 剥一个常见后缀；剥后剩余 < 2 字符则不动。
    static func stripSuffix(_ w: String) -> String {
        for s in SUFFIXES {
            if w.hasSuffix(s) && w.count - s.count >= 2 {
                return String(w.dropLast(s.count))
            }
        }
        return w
    }

    /// 0..1 相似度，对常见词形变化友好。
    static func similarity(_ q: String, _ h: String) -> Double {
        if q == h { return 1.0 }
        let qs = stripSuffix(q)
        let hs = stripSuffix(h)
        if !qs.isEmpty && qs == hs { return 0.95 }
        // query 是 head 的短前缀（截断场景：head 越长越像截断 → 一点加成）
        if h.hasPrefix(q) {
            let d = h.count - q.count
            if d > 0 && d <= 3 { return 0.9 + 0.05 * Double(d) / 3.0 }
        }
        // head 是 query 的短前缀（query 后面多带了尾巴）
        if q.hasPrefix(h) {
            let d = q.count - h.count
            if d > 0 && d <= 3 { return 0.9 }
        }
        let dist = Double(levenshtein(q, h))
        let m = Double(max(q.count, h.count))
        return 1.0 - dist / max(m, 1)
    }

    /// 返回按相似度降序、score > 0 的前 k 个候选。
    static func suggest(query: String, heads: [String], k: Int = 10) -> [(score: Double, head: String)] {
        let q = query.lowercased()
        let scored: [(Double, String)] = heads.map { (similarity(q, $0.lowercased()), $0) }
        let sorted = scored.sorted { $0.0 > $1.0 }
        return sorted.prefix(k).filter { $0.0 > 0 }.map { (score: $0.0, head: $0.1) }
    }
}


// MARK: - Format ─────────────────────────────────────────────────────────────

/// 把 JSON 行渲染为可读字符串。格式与 scripts/cet4.py format_detail 对齐。
enum Format {

    static func detail(_ row: [String: Any]) -> String {
        let word = _safe(row, "content", "word") as? [String: Any] ?? [:]
        let c = word["content"] as? [String: Any] ?? [:]
        let head = (word["wordHead"] as? String) ?? "?"
        let wordId = (word["wordId"] as? String) ?? ""
        let usphone = (c["usphone"] as? String) ?? ""
        let ukphone = (c["ukphone"] as? String) ?? ""
        let phone = (c["phone"] as? String) ?? ""

        var lines: [String] = []

        // 标题行
        let phones = [ukphone, usphone, phone].filter { !$0.isEmpty }
        let title = phones.isEmpty ? head : "\(head)  " + phones.map { "/\($0)/" }.joined(separator: "  ")
        lines.append(title)
        if !wordId.isEmpty {
            lines.append("# \(wordId)")
        }
        lines.append(String(repeating: "─", count: max(40, title.count)))

        // 释义（中英双语：中文 + 缩进的"英"行）
        if let trans = c["trans"] as? [[String: Any]], !trans.isEmpty {
            lines.append("【释义】")
            for t in trans {
                let pos = (t["pos"] as? String) ?? ""
                let cn = (t["tranCn"] as? String) ?? ""
                let other = (t["tranOther"] as? String) ?? ""
                if pos.isEmpty && cn.isEmpty && other.isEmpty { continue }
                let head_txt = pos.isEmpty ? "" : pos
                let cnLine = cn.isEmpty
                    ? "  \(head_txt)".trimmingCharacters(in: .whitespaces)
                    : "  \(head_txt) \(cn)".trimmingCharacters(in: CharacterSet(charactersIn: " "))
                if !cnLine.isEmpty { lines.append(cnLine) }
                if !other.isEmpty { lines.append("    英 \(other)") }
            }
        }

        // 例句
        if let sents = (_safe(c, "sentence", "sentences") as? [[String: Any]]), !sents.isEmpty {
            lines.append("【例句】")
            for (i, s) in sents.enumerated() {
                let en = (s["sContent"] as? String) ?? ""
                let cn = (s["sCn"] as? String) ?? ""
                lines.append("  \(i + 1). \(en)")
                if !cn.isEmpty { lines.append("     \(cn)") }
            }
        }

        // 真题例句
        if let exams = (_safe(c, "realExamSentence", "sentences") as? [[String: Any]]), !exams.isEmpty {
            lines.append("【真题】")
            for s in exams {
                let en = (s["sContent"] as? String) ?? ""
                let info = (_safe(s, "sourceInfo") as? [String: Any]) ?? [:]
                let tag = [info["year"], info["paper"], info["type"]]
                    .compactMap { $0 }
                    .map { "\($0)" }
                    .joined(separator: " · ")
                if !tag.isEmpty {
                    lines.append("  [\(tag)] \(en)")
                } else {
                    lines.append("  \(en)")
                }
            }
        }

        // 短语
        if let phrases = (_safe(c, "phrase", "phrases") as? [[String: Any]]), !phrases.isEmpty {
            lines.append("【短语】")
            for p in phrases {
                let en = (p["pContent"] as? String) ?? ""
                let cn = (p["pCn"] as? String) ?? ""
                lines.append("  \(en) — \(cn)")
            }
        }

        // 同义词
        if let synos = (_safe(c, "syno", "synos") as? [[String: Any]]), !synos.isEmpty {
            lines.append("【同义词】")
            for s in synos {
                let pos = (s["pos"] as? String) ?? ""
                let tran = (s["tran"] as? String) ?? ""
                let words = ((s["hwds"] as? [[String: Any]]) ?? [])
                    .compactMap { $0["w"] as? String }
                let head = [pos, tran].filter { !$0.isEmpty }.joined(separator: " ")
                if !head.isEmpty && !words.isEmpty {
                    lines.append("  \(head): \(words.joined(separator: ", "))")
                } else if !words.isEmpty {
                    lines.append("  \(words.joined(separator: ", "))")
                }
            }
        }

        // 反义词
        if let antos = (_safe(c, "antos", "anto") as? [[String: Any]]), !antos.isEmpty {
            let items = antos.compactMap { $0["hwd"] as? String }.filter { !$0.isEmpty }
            if !items.isEmpty {
                lines.append("【反义词】 " + items.joined(separator: ", "))
            }
        }

        // 词根相关
        if let rels = (_safe(c, "relWord", "rels") as? [[String: Any]]), !rels.isEmpty {
            lines.append("【词根相关】")
            for r in rels {
                let pos = (r["pos"] as? String) ?? ""
                let words = (r["words"] as? [[String: Any]]) ?? []
                let parts = words.compactMap { w -> String? in
                    guard let h = w["hwd"] as? String, !h.isEmpty else { return nil }
                    let tran = (w["tran"] as? String) ?? ""
                    return tran.isEmpty ? h : "\(h) (\(tran))"
                }
                if parts.isEmpty { continue }
                let prefix = pos.isEmpty ? "" : "\(pos): "
                lines.append("  \(prefix)" + parts.joined(separator: "; "))
            }
        }

        // 记忆法
        if let rem = (_safe(c, "remMethod", "val") as? String), !rem.isEmpty {
            lines.append("【记忆】")
            lines.append("  \(rem)")
        }

        return lines.joined(separator: "\n")
    }

    /// 安全取嵌套字段；找不到返回 default（nil 表示不存在）。
    private static func _safe(_ obj: Any?, _ keys: String...) -> Any? {
        var cur: Any? = obj
        for k in keys {
            guard let d = cur as? [String: Any], d[k] != nil else { return nil }
            cur = d[k]
        }
        return cur
    }
}


// MARK: - VocabWindow ───────────────────────────────────────────────────────

/// 主窗口控制器：搜索框 + 随机按钮 + 详情文本视图。
class VocabWindow: NSObject, NSWindowDelegate, NSTextViewDelegate {
    let index: Index
    let window: NSWindow
    let searchField: NSSearchField
    let randomButton: NSButton
    let scrollView: NSScrollView
    let textView: NSTextView

    init(index: Index) {
        self.index = index

        // 窗口
        let style: NSWindow.StyleMask = [.titled, .closable, .miniaturizable, .resizable]
        let win = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 600, height: 800),
            styleMask: style,
            backing: .buffered,
            defer: false
        )
        win.title = "Vocab Viewer"
        win.minSize = NSSize(width: 420, height: 360)
        win.center()
        self.window = win

        // 搜索框（target/action 留到 super.init 之后设置）
        let sf = NSSearchField()
        sf.placeholderString = "Search a word…"
        self.searchField = sf

        // 随机按钮
        let btn = NSButton(title: "Random", target: nil, action: nil)
        btn.bezelStyle = .rounded
        self.randomButton = btn

        // 文本视图
        let scroll = NSScrollView()
        scroll.hasVerticalScroller = true
        scroll.hasHorizontalScroller = false
        scroll.autohidesScrollers = false
        scroll.borderType = .noBorder
        scroll.drawsBackground = true
        scroll.backgroundColor = .textBackgroundColor
        self.scrollView = scroll

        let tv = NSTextView()
        tv.isEditable = true
        tv.isSelectable = true
        tv.isRichText = false
        tv.allowsUndo = true
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isAutomaticSpellingCorrectionEnabled = false
        tv.usesFindBar = true
        tv.isVerticallyResizable = true
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.containerSize = NSSize(
            width: scroll.contentSize.width,
            height: CGFloat.greatestFiniteMagnitude
        )
        tv.font = NSFont.userFixedPitchFont(ofSize: 13)
            ?? NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        tv.textContainerInset = NSSize(width: 8, height: 8)
        self.textView = tv
        scroll.documentView = tv

        super.init()

        // super.init 之后才能把自己设为 target/delegate
        searchField.target = self
        searchField.action = #selector(searchSubmitted(_:))
        randomButton.target = self
        randomButton.action = #selector(showRandom)
        textView.delegate = self
        window.delegate = self

        installContent()
    }

    /// 顶部控件 + 文本视图，frame-based 布局（不依赖 storyboard）
    private func installContent() {
        let content = NSView(frame: window.contentView!.bounds)
        content.autoresizingMask = [.width, .height]
        content.wantsLayer = true
        content.layer?.backgroundColor = NSColor.windowBackgroundColor.cgColor
        window.contentView = content

        let topHeight: CGFloat = 40
        let pad: CGFloat = 8

        searchField.frame = NSRect(
            x: pad, y: content.bounds.height - topHeight,
            width: content.bounds.width - 110 - pad * 2,
            height: 24
        )
        searchField.autoresizingMask = [.width, .minYMargin]

        randomButton.frame = NSRect(
            x: content.bounds.width - 100 - pad, y: content.bounds.height - topHeight - 1,
            width: 92, height: 27
        )
        randomButton.autoresizingMask = [.minXMargin, .minYMargin]

        scrollView.frame = NSRect(
            x: 0, y: 0,
            width: content.bounds.width, height: content.bounds.height - topHeight
        )
        scrollView.autoresizingMask = [.width, .height]

        // 让 textView 的 textContainer 跟上 scrollView 宽度
        textView.minSize = NSSize(width: 0, height: 0)
        textView.maxSize = NSSize(
            width: CGFloat.greatestFiniteMagnitude,
            height: CGFloat.greatestFiniteMagnitude
        )
        textView.frame = scrollView.bounds
        textView.autoresizingMask = [.width]

        content.addSubview(searchField)
        content.addSubview(randomButton)
        content.addSubview(scrollView)
    }

    // MARK: 动作

    @objc func showRandom() {
        render(index.randomHead())
    }

    /// 搜索框回车 / 点搜索按钮
    @objc func searchSubmitted(_ sender: NSSearchField) {
        let q = (sender.stringValue).trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return }
        // 优先精确匹配；找不到给候选
        if let off = index.offsets[q] {
            render(q, sourceOffset: off)
        } else {
            renderSuggestions(for: q)
        }
    }

    /// ⌘F：聚焦搜索框
    @objc func focusSearch() {
        window.makeFirstResponder(searchField)
        if let editor = searchField.currentEditor() {
            editor.selectedRange = NSRange(location: searchField.stringValue.count, length: 0)
        }
    }

    /// ⌘N：显示随机词（菜单调用）
    @objc func newRandom() {
        showRandom()
    }

    // MARK: 渲染

    private func render(_ head: String, sourceOffset: Int? = nil) {
        do {
            let offset = sourceOffset ?? (index.offsets[head] ?? -1)
            guard offset >= 0 else { renderSuggestions(for: head); return }
            let row = try index.fetch(offset)
            let text = Format.detail(row)
            setText(text)
        } catch {
            setText("读取失败：\(error.localizedDescription)")
        }
    }

    private func renderSuggestions(for query: String) {
        let sgs = Suggest.suggest(query: query, heads: index.heads, k: 10)
        var out: [String] = []
        out.append("未找到 \"\(query)\"。你可能想找:")
        if sgs.isEmpty {
            out.append("  （无相近候选）")
        } else {
            for s in sgs {
                out.append(String(format: "  - %-18@ (%.2f)", s.head as NSString, s.score))
            }
        }
        setText(out.joined(separator: "\n"))
    }

    private func setText(_ text: String) {
        textView.string = text
        textView.scroll(NSPoint.zero)
    }

    // MARK: NSWindowDelegate

    func windowDidBecomeKey(_ notification: Notification) {
        // 首次成 key 时让 textView 获得焦点，便于 ⌘C
        if window.firstResponder == nil {
            window.makeFirstResponder(textView)
        }
    }
}


// MARK: - Main Menu ──────────────────────────────────────────────────────────

/// 装主菜单。**Edit 子菜单必装**，否则 NSTextView 的 ⌘C/⌘V/⌘A/⌘Z 全部失灵。
func installMainMenu(window: VocabWindow) {
    let mainMenu = NSMenu()

    // ── App 菜单 ──
    let appItem = NSMenuItem()
    mainMenu.addItem(appItem)
    let appMenu = NSMenu()
    appItem.submenu = appMenu
    appMenu.addItem(
        withTitle: "About VocabViewer",
        action: #selector(NSApplication.orderFrontStandardAboutPanel(_:)),
        keyEquivalent: ""
    )
    appMenu.addItem(.separator())
    let quit = NSMenuItem(
        title: "Quit VocabViewer",
        action: #selector(NSApplication.terminate(_:)),
        keyEquivalent: "q"
    )
    quit.keyEquivalentModifierMask = [.command]
    appMenu.addItem(quit)

    // ── Edit 菜单（必装）──
    let editItem = NSMenuItem()
    mainMenu.addItem(editItem)
    let editMenu = NSMenu(title: "Edit")
    editItem.submenu = editMenu

    // Undo / Redo 转发到 firstResponder.undoManager
    editMenu.addItem(
        withTitle: "Undo",
        action: Selector(("undo:")),
        keyEquivalent: "z"
    )
    editMenu.addItem(
        withTitle: "Redo",
        action: Selector(("redo:")),
        keyEquivalent: "Z"
    )
    editMenu.addItem(.separator())
    // Cut / Copy / Paste / Select All 用 NSText/NSTextView 自带 responder action
    editMenu.addItem(
        withTitle: "Cut",
        action: #selector(NSText.cut(_:)),
        keyEquivalent: "x"
    )
    editMenu.addItem(
        withTitle: "Copy",
        action: #selector(NSText.copy(_:)),
        keyEquivalent: "c"
    )
    editMenu.addItem(
        withTitle: "Paste",
        action: #selector(NSText.paste(_:)),
        keyEquivalent: "v"
    )
    editMenu.addItem(
        withTitle: "Select All",
        action: #selector(NSStandardKeyBindingResponding.selectAll(_:)),
        keyEquivalent: "a"
    )

    // ── View 菜单 ──
    let viewItem = NSMenuItem()
    mainMenu.addItem(viewItem)
    let viewMenu = NSMenu(title: "View")
    viewItem.submenu = viewMenu
    let focusItem = NSMenuItem(
        title: "Focus Search",
        action: #selector(VocabWindow.focusSearch),
        keyEquivalent: "f"
    )
    focusItem.keyEquivalentModifierMask = [.command]
    focusItem.target = window
    viewMenu.addItem(focusItem)

    // ── Word 菜单 ──
    let wordItem = NSMenuItem()
    mainMenu.addItem(wordItem)
    let wordMenu = NSMenu(title: "Word")
    wordItem.submenu = wordMenu
    let newItem = NSMenuItem(
        title: "New Random Word",
        action: #selector(VocabWindow.newRandom),
        keyEquivalent: "n"
    )
    newItem.keyEquivalentModifierMask = [.command]
    newItem.target = window
    wordMenu.addItem(newItem)

    // ── Window 菜单（标准）──
    let winItem = NSMenuItem()
    mainMenu.addItem(winItem)
    let winMenu = NSMenu(title: "Window")
    winItem.submenu = winMenu
    winMenu.addItem(
        withTitle: "Minimize",
        action: #selector(NSWindow.performMiniaturize(_:)),
        keyEquivalent: "m"
    )
    winMenu.addItem(
        withTitle: "Zoom",
        action: #selector(NSWindow.performZoom(_:)),
        keyEquivalent: ""
    )

    NSApp.mainMenu = mainMenu
}


// MARK: - AppDelegate ────────────────────────────────────────────────────────

class AppDelegate: NSObject, NSApplicationDelegate {
    var window: VocabWindow!

    func applicationDidFinishLaunching(_ notification: Notification) {
        window.window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        window.showRandom()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}


// MARK: - Entry Point ────────────────────────────────────────────────────────

/// 按优先级找包含 vocab_all.idx 的目录。
/// - 如果用户显式给了 `--data <dir>`，只在那个目录找；找不到直接返回 nil。
/// - 否则按顺序：二进制同目录 → 当前工作目录。
func resolveDataDir(cliData: String?) -> URL? {
    let candidates: [URL]
    if let cli = cliData {
        candidates = [URL(fileURLWithPath: cli)]
    } else {
        candidates = [
            URL(fileURLWithPath: Bundle.main.bundlePath),
            URL(fileURLWithPath: FileManager.default.currentDirectoryPath),
        ]
    }
    for dir in candidates {
        let idx = dir.appendingPathComponent("vocab_all.idx")
        if FileManager.default.fileExists(atPath: idx.path) {
            return dir
        }
    }
    return nil
}

/// 解析 CLI 参数：只支持 --data <dir>
func parseCLIArgs() -> String? {
    let args = CommandLine.arguments
    var i = 1
    while i < args.count {
        if args[i] == "--data" && i + 1 < args.count {
            return args[i + 1]
        }
        i += 1
    }
    return nil
}

func main() {
    let cliData = parseCLIArgs()
    guard let dataDir = resolveDataDir(cliData: cliData) else {
        FileHandle.standardError.write(Data(
            "error: 未找到 vocab_all.idx。请确认数据目录，或先运行：\n".utf8
        ))
        FileHandle.standardError.write(Data(
            "    python3 scripts/build-vocab.py\n".utf8
        ))
        exit(1)
    }

    let index: Index
    do {
        index = try Index.load(jsonDir: dataDir)
    } catch {
        FileHandle.standardError.write(Data(
            "error: 加载索引失败：\(error.localizedDescription)\n".utf8
        ))
        exit(1)
    }

    let app = NSApplication.shared
    let delegate = AppDelegate()
    app.delegate = delegate
    let window = VocabWindow(index: index)
    delegate.window = window
    installMainMenu(window: window)

    app.setActivationPolicy(.regular)
    app.run()
}

main()
