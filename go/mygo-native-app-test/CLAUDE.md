# CLAUDE.md

MyGo 原生桌面应用：本地 Ollama 翻译。**纯原生 `ui` 组件，没有 webview、没有前端页面。**

- MyGo `v0.7.4`，Go `1.27.3`
- 源码位置：`go list -m -f '{{.Dir}}' github.com/egoist/mygo` → `~/go/pkg/mod/github.com/egoist/mygo@v0.7.4`
- 改 API 前先读 `ui/doc.go`；`ui/api_gen.go` 是生成的公开 API（约 6500 行，元素方法都在里面，接收者名被生成器改成 `_handle`，用 `grep "func (_handle Element)"` 找）
- 官方文档在 `<module>/docs/ui/*.md`（`views.md` `testing.md` `text-input.md` `select.md` `spinner.md` `toast.md` 最相关）
- `.agents/skills/mygo-maintenance/SKILL.md` 是这个项目自带的维护 skill，动手前读它

## 命令

```sh
go tool mygo dev            # 开发构建，自动重载
go tool mygo build          # 产出 .app 和 .dmg
go tool mygo vet .          # 检查原生 UI 的生命周期违规（存 Element/Context、goroutine 捕获）
go tool mygo doctor         # 检查环境（go/bun/macOS SDK）
CGO_ENABLED=0 go test ./... # MyGo 要求无 cgo
go test -race ./...         # 异步部分必须过 race
gofmt -l . && go vet ./...
```

## MyGo 原生 UI：核实过的机制

下面每条都对着 `v0.7.4` 源码读过，不是猜的。**改这块之前如果版本升级了，重新核对。**

### 一帧是怎么跑的

`ui/runtime.go` 的 `runFrame`：

```
for pass := 0; pass < 3; pass++ {
    rt.view(&rt.c)        // 构造
    rt.applyAllInputs()   // 输入
    rt.runNoticeActions() // OnChange / OnSubmit
    rt.runActions()       // OnClick / OnShortcut
    rt.endPass()          // forgetInput：清掉这一轮控件的 clicks
    if !rt.consumed { break }   // 有动作被处理就再来一轮
}
```

要点：

- **一次点击只触发一次动作**。`endPass` → `forgetInput` 把 `s.clicks` 清零，所以第二轮 pass 里 `Clicked()` 已经是 false。否则每帧都会重复触发。
- **`OnClick` 在 `inFrame` 为 true 时执行**（`rt.inFrame = true` 在 537 行，`= false` 在 1102 行）。所以 `OnClick` 里可以调 `c.WriteClipboard` / `c.Toast` / `c.OpenURL`，这些用 `c.build()` 判断有效性的方法都不会被跳过。
- **有动作就重画，最多 3 轮**。所以 `OnClick` 里改绑定值，同一帧内就能看到效果，不需要额外请求重绘。

### 输入的时间点

- 控件在构造函数返回时就拿到了这轮输入（像 Dear ImGui/egui）。
- `OnClick` / `OnShortcut` 在构造**之后**、绑定值更新**之后**跑。
- **要回写绑定值就用 `OnClick`，不要用 `Clicked()`**。本项目的「⇄ 交换语言」就是靠这个：`a.from` / `a.to` 是 `ui.Select` 绑定的值，`OnClick` 里交换它们，下一轮 pass 两个下拉就显示新值了。

### 身份与生命周期

- `c.Key(key)` 给**下一个**控件固定身份，必须在构造控件**之前**给：`ui.TextArea(c.Key("source"), &a.source)`。有状态的控件都要给，key 唯一。
- `Element.Key(key)` 是给容器用的，且要在子控件之前。
- Element 跨帧就失效（同一帧内换一轮 pass 也失效）。**永远不要把 Element 存进结构体**，也不要跨帧使用。`ui.Handle` 才是持久身份。
- **不要把 `*ui.Context` 存进结构体或捕获进 goroutine**。`c.Services()` 是给持久回调用的（剪贴板、URL、后台重绘），它不持有窗口。

### 异步

- goroutine 里干活，结果用 `mygo.Window.Update(fn)` 回主线程 —— 它在主线程执行 `fn` 然后重绘。
- `Window.Invalidate()` 只重绘，不改状态。改了模型字段就要用 `Update`。
- **goroutine 里绝对不要读界面状态**。本项目踩过：`go func(){ a.doTranslate(a.from, a.to, text) }()` 在用户翻译途中点「⇄」时会竞争。正确写法是点击时就把值取成局部变量：

  ```go
  text, from, to := a.source, a.from, a.to   // 主线程取好
  go func() {
      out, err := a.doTranslate(from, to, text)
      a.publish(func() { /* 主线程改状态 */ })
  }()
  ```

- App 退出后 `Window.Update` 的回调会被丢弃（`loop.go` 的 `postMain` 返回 false），不会 panic。

### tester 找元素的规则（`ui/runtime.go` 的 `rt.labels`）

这一段最容易踩。注册进 `rt.labels` 的文字按这个优先级：

1. 控件的 `Label(...)` —— **最高优先级**
2. `Form` 里的控件：父级 `Field` 的 Label
3. 其它：元素的 `e.text`（普通 `ui.Text` 的文字）

**输入框（`TextInput`/`TextArea`）的值存在 `st.editor` 里，`e.text` 是空的。** 所以：

- 输入框**必须给 `Label`**，否则 tester 根本找不到它（空值时连注册都不注册）。
- **输入框里的内容 tester 看不见**：`tt.Texts()` 拿不到译文，断言要落在模型字段上。
- `tt.Find(s)` / `tt.Click(s)` 是**精确匹配**，`tt.HasText(s)` 是**包含匹配**。
  本项目标题叫「本地翻译」而按钮叫「翻译」，两者不冲突，就是因为精确匹配。
  反过来，同名的两个控件 `tt.Click` 只会点到第一个。

### 其它已核实的 API

- `ui.TextArea(c, *string)` 多行；`Lines(min, max)` 限行数，超了内部滚动。
- `ui.Select(c, *string, options []string)`：选中值必须**就是** options 里的字符串，下拉显示的就是选中值本身。
- `ui.Spinner(c)` 不定长进度；`ui.Progress(c, v)` 传负数表示「长度未知」。
- `ui.Context.OnShortcut(mods, key, fn)` 声明窗口级快捷键，写在 view 顶层（`Children` 回调外面）。
  **`ui.Cmd` 在 macOS 上是 Command、在别处是 Control**；`ui.Super` 是平台原始的 Command/Windows 键。
  （`ui` 只导出 `Shift` / `Ctrl` / `Alt` / `Super` / `Cmd`，我一开始 grep `ModCmd` 没找到就以为 macOS 没有 Command —— 是我名字猜错了。）
- `ui.Render(view, w, h, scale)` 返回 `*image.RGBA`：**不用截屏权限就能看到界面长什么样**，改布局时比开 app 看快。

### 窗口尺寸会被存档覆盖

`WindowOptions.StateKey` 会记住窗口上次的位置和大小，存在
`~/Library/Application Support/<app 名>/window-state.json`。**改了 `Width`/`Height` 但窗口没变，就是被存档覆盖了**，删掉那个文件或用新的 `StateKey` 才会生效。开发版是 `mygo-native-app-test Dev` 目录，和正式版分开。

### 驱动真正的 app：automation 插件

`ui.NewTester` 只在 `go test` 里跑。**要驱动打包后的 app，用 MyGo 的 automation 插件**
（`<module>/docs/plugins/automation.md`）：

```go
//go:build e2e
package main
import _ "github.com/egoist/mygo/plugins/automation"
```

```sh
go build -tags e2e -o /tmp/mygo-e2e .
MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e &
python3 e2e/drive.py        # 本项目的驱动脚本
```

协议是 Unix socket 上的换行分隔 JSON。常用方法：

| 方法 | 作用 |
|---|---|
| `windows` | 列出窗口 |
| `tree` | 窗口里辅助技术能读到的元素（按 `key` 认） |
| `act` | 按 `key` 对元素做动作：`press` `focus` `setValue` `insertText` |
| `input` | 发指针/键盘/文字事件（坐标是窗口内容里的 DIP） |
| `capture` | 截窗口的内容，返回 PNG |
| `idle` | 等窗口把要画的帧画完 |

**要在元素上给 `key`，不然只能按 id 找**（id 每帧可能变）。本项目给翻译按钮、复制按钮、
两个下拉、源文本框、译文框都给了 key。

**两个坑，都花了我不少时间：**

1. **系统合成的键盘鼠标事件进不去 MyGo 的原生 UI 窗口。** 用 AppleScript
   (`System Events` 的 `click at` / `keystroke`) 点和打都无效 —— 窗口是自己画的，
   不走 AppKit 的控件事件。**用 automation 插件的 `act` / `input`**，它直接走 app 自己的事件通道。
2. **窗口被挡住的时候不画帧。** `tree` 报的是**最后一帧**里的元素，所以窗口在别的窗口后面时，
   app 改了状态、也调了 `Window.Update`，但没有新帧被画出来，`tree` 就一直是旧的 ——
   看起来像「结果永远不回来」。
   诊断时我为此花了不少时间：加了日志确认 publish 确实执行了、`Window.Update` 也确实调了，
   最后才发现是遮挡。
   **解法**：读 `tree` 之前先 `capture` 一次 —— 它会强制渲染一帧，树就跟上来了；
   另外把窗口置前（`osascript` 设 `frontmost`）让人眼也能看到。

**完整做法、协议细节和排查手册见 [docs/e2e-testing.md](docs/e2e-testing.md)。**
## 本项目的探索过程

按踩坑顺序记下来，省得下次重走：

1. **先摸清模型行为再写界面。** 用 `curl` 直接打 `http://localhost:11434/api/generate` 试了几种 prompt 格式：
   - `Translate from Chinese to English: 我喜欢人工智能。` → `I like artificial intelligence.` ✅
   - 不带前缀只给原文 → 模型当成聊天，续写了一大段介绍 AI 的中文 ❌
   - 中文→日文、英文→法文、中文→韩文都正常。

   **结论：`Translate from {源} to {目标}: {文本}` 这个前缀是硬约束**，丢了模型就完全不翻译。`translate_test.go` 里把 prompt 格式断言死了，防止以后改坏。

2. **语言名要用英文。** 下拉里显示「中文」，但送给模型的是 `Chinese`。模型只认英文指令。`translate.go` 里 `lang{label, name}` 两个字段分开，`langName()` 查表。

3. **本机设了代理。** `http_proxy=http://127.0.0.1:10808`。Go 默认不代理 loopback，但 HTTP client 还是显式写了 `Transport: &http.Transport{Proxy: nil}`，不靠运气。

4. **异步的两次竞争。** race detector 报了两次，两次都是真 bug：
   - 第一次在**测试夹具**里：测试的 `publish` 直接在 goroutine 里执行状态变更，而真实 app 是 `Window.Update` 回主线程。改成测试把变更函数交给测试 goroutine 执行。
   - 第二次在**业务代码**里：goroutine 读 `a.from`/`a.to`，用户在翻译途中点「⇄」就会写它们。改成点击时取局部变量。
     `TestViewSwapWhileTranslating` 是对着这个 bug 写的：把修复改回去，race detector 会精确指向 swap 的写和 goroutine 的读。

5. **界面靠 `ui.Render` 出图检查。** 不用截屏（本机终端没有录屏权限，`screencapture` 直接失败）。`ui.Render` 渲染成 PNG 看：成功态、错误态、加载态各一张。第一版窗口 560×720 出来下半屏全空，收成 560×520。README 顶上那张图是 `screenshot_test.go` 生成的，`RENDER_PNG=1 go test -run TestScreenshot` 重来。

6. **真机验证靠 automation 插件，不靠 AppleScript。** `e2e/drive.py` 驱动打包后的 app：填文本、按翻译、读元素树、截图、验剪贴板。两个方向（中→英、英→中）和复制按钮都验过。
   排查「结果不回来」时先确认三件事：请求发出去了吗（看 ollama）、publish 跑了吗（加日志）、帧画出来了吗（`capture` 一次）。
   这次就是卡在第三件：**窗口被终端挡住 → 不画帧 → 元素树停在旧的一帧**。

7. **错误提示要写成人话。** 连不上 ollama 报「无法连接 Ollama（http://localhost:11434），请确认 ollama serve 已启动」，而不是 `dial tcp ... connection refused`。模型返回 `error` 字段时直接透出。

8. **`mygo vet` 查的是原生 UI 的生命周期违规**（存 Element/Context、goroutine 捕获），不是普通 vet 的全部内容。两个都跑。

## 改这个项目时注意

- 界面文案是中文，注释也是中文；变量名、函数名是英文。
- `app` 结构体只有数据字段和两个函数指针（`doTranslate` 给测试替换、`publish` 给测试控制发布时机）。**不要再往里塞 Element 或 Context。**
- `newApp()` 里设默认值和 `publish`；测试用 `newApp()` 拿一个干净的实例。
- 窗口标题是「MyGo 翻译」，`mygo.json` 里 `name` 还是 `mygo-native-app-test`（打包名）。
- ⌘↩ 快捷键声明在 `view` 的最外层（`Children` 回调外面）：`c.OnShortcut(ui.Cmd, ui.KeyEnter, a.translate)`。
  放在 `Children` 回调里会被算成父作用域的局部快捷键，不是窗口级的。界面上的提示文字由 `translateHint()` 生成，
  按平台显示「⌘↩ 翻译」或「Ctrl+↩ 翻译」。
