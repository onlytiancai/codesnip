# 端到端测试：驱动真正的 app

本文讲怎么测试**打包后的** MyGo 应用 —— 真的窗口、真的点击、真的模型响应。
单元测试（`ui.NewTester`）只跑视图，这一层跑的是整个 app。

MyGo 自带 [automation 插件](https://github.com/egoist/mygo/blob/main/docs/plugins/automation.md)，
用 Unix socket 上的换行分隔 JSON 协议，让另一个进程像用户一样操作 app。

---

## 两种测试，别搞混

| | `ui.NewTester` | automation 插件 |
|---|---|---|
| 跑在哪 | `go test` 进程内 | 另一个进程，连 socket |
| 看到什么 | 只有视图 | 真正的窗口、辅助技术看到的元素、菜单、剪贴板 |
| 能截图 | `ui.Render` / `tt.Image()` | `capture`（窗口的真实内容，PNG） |
| 要真模型 | 不要 | 要（本项目要 ollama 起着） |
| 适合 | 逻辑、状态机、错误分支 | 真窗口、快捷键、剪贴板、跨进程 |

**先用 `ui.NewTester`** —— 快、不用起进程、不用模型。逻辑都对了，再用 automation 验真机。

---

## 1. 打开插件

插件只能被 app **引入**才会链接进去，所以单独一个文件、用构建标签关住：

```go
//go:build e2e

package main

// automation 插件让另一个进程驱动这个 app：读它的窗口和元素、往里输入、
// 按它的按钮、把它截成图。只在 -tags e2e 的构建里有，正式包不受影响。
//
//	go build -tags e2e -o /tmp/mygo-e2e .
//	MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e
import _ "github.com/egoist/mygo/plugins/automation"
```

它**只在 `MYGO_AUTOMATION` 指到一个路径时才监听**，否则什么都不做。

> 注意：`mygo build` 的正式构建不带 inspector（相当于 `-tags mygo_noinspector`），那个构建**不含**插件，
> 即使 import 了也会忽略 `MYGO_AUTOMATION` 并在 stderr 说明（`-debug` 或 `MYGO_INSPECTOR=1` 保留 inspector 时除外）。
> 所以 e2e 要自己 `go build -tags e2e`。

## 2. 构建并运行

```sh
go build -tags e2e -o /tmp/mygo-e2e .
MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e &
```

socket 放在只有当前用户能写的目录里（macOS/Linux 上是 `0600` 权限）。
app 退出时会删掉它；上次崩了留下的会被替换，但**别人占着的 socket 不会被抢**。

## 3. 给要操作的元素起 key

automation 按 **`key`** 找元素。id 每帧可能变，key 是 app 自己给的名字，不会变。

```go
// 有状态的控件：key 必须在构造之前给
ui.TextArea(c.Key("source"), &a.source).Label("源文本")
ui.Select(c.Key("from"), &a.from, langLabels).Label("源语言")
ui.PrimaryButton(c.Key("translate"), "翻译")
```

`c.Key(k)` 给**下一个**控件；`Element.Key(k)` 给容器，且要在子控件之前。
只给测试要碰的元素起 key 就好 —— 布局容器不需要出现在辅助技术的树里。

## 4. 协议

任何能开 Unix socket 的语言都能当客户端。

```
→ {"id": 1, "method": "tree", "params": {"window": 1}}
← {"id": 1, "result": {"window": 1, "nodes": [...]}}
```

每个请求一行 JSON，每个回答一行，**按顺序**、一个答完才答下一个。
要并发就多开几条连接。

回答有 `result`，或者有 `error`（带 `code` 和 `message`）：

| code | 意思 |
|---|---|
| `parse_error` | 这行不是 JSON 对象 |
| `invalid_request` | 没有 method，或参数用不了 |
| `unknown_method` | 没有这个方法 |
| `no_window` | 没有这个 id 的窗口，或 app 一个窗口都没有 |
| `no_element` | 窗口最后一帧里没有这个 id 或 key 的元素 |
| `unsupported` | 这个窗口做不了（如给网页窗口读元素） |
| `timeout` | `idle` 等超时了 |
| `failed` | app 做不到（如脚本报错） |

### 方法

除 `windows` 外都要 `window`（不给就是当前聚焦的窗口，再没有就是第一个）。
坐标和尺寸是**窗口内容里的 DIP**，和元素的 bounds 一样。

| 方法 | 参数 | 作用 |
|---|---|---|
| `windows` | | 列出窗口：`{id, title, kind, bounds, contentBounds, focused, visible}` |
| `tree` | `window` | 窗口里辅助技术能读到的元素 |
| `act` | `window`, `key` 或 `id`, `action`, `text?` | 对元素做动作 |
| `input` | `window`, `events` | 发指针/滚轮/键盘/文字事件 |
| `menu` | `window`, `path` | 按路径选菜单项，如 `["File", "New Window"]` |
| `capture` | `window` | 窗口内容，返回 base64 的 PNG |
| `eval` | `window`, `script` | 在网页窗口里跑 JavaScript |
| `idle` | `window`, `timeout?` | 等窗口画完它要的帧 |

### `act` 的动作

| 动作 | 作用 |
|---|---|
| `press` | 按按钮、勾选复选框、打开菜单按钮 |
| `focus` | 给元素键盘焦点 |
| `setValue` | 换掉文本框的内容（`text`） |
| `insertText` | 在文本框的光标处插入（`text`） |
| `selectText` | 选中文本框的某一段（`from`, `to`） |
| `expand` / `collapse` | 展开/收起树节点 |
| `increment` / `decrement` | 加减滑块、步进器 |
| `scrollIntoView` | 滚动到元素露出来 |

### `input` 的事件

`{"type": "click", "x": 200, "y": 99}` 这种。

| type | 字段 |
|---|---|
| `pointerMove` | `x`, `y`, `mods?` |
| `pointerDown`, `pointerUp` | `x`, `y`, `button?`, `clicks?`, `mods?` |
| `click` | `x`, `y`, `button?`, `clicks?`, `mods?`：一次移动 + 一次按下 + 一次抬起 |
| `wheel` | `x`, `y`, `dx`, `dy`, `precise?`, `mods?` |
| `keyDown`, `keyUp`, `key` | `key`：键名和修饰键，写法跟菜单快捷键一样（`Enter`、`Shift+Tab`、`CmdOrCtrl+Z`） |
| `text` | `text`：输入的文字，或输入法提交的 |

`mods` 可以是 `shift`、`ctrl`、`alt`、`super`、`cmdOrCtrl`。
**按键事件不带文字**：要打出的文字另发一个 `text` 事件，像后端做的那样。

`button` 是 `left`（默认）、`right`、`middle`。

## 5. 一个完整的例子

本项目的 `e2e/drive.py` 就是这些东西拼起来的（Python 只用标准库）：

```python
import base64, json, socket, subprocess, time

SOCK = "/tmp/mygo-e2e.sock"

class Client:
    def __init__(self, path=SOCK):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(path)
        self.f = self.sock.makefile("rwb")
        self.id = 0

    def call(self, method, params=None):
        self.id += 1
        req = {"id": self.id, "method": method}
        if params:
            req["params"] = params
        self.f.write((json.dumps(req) + "\n").encode())
        self.f.flush()
        ans = json.loads(self.f.readline())
        if "error" in ans:
            raise SystemExit(f"{method} failed: {ans['error']}")
        return ans["result"]

c = Client()
win = c.call("windows")["windows"][0]["id"]

def state(refresh=False):
    if refresh:
        c.call("capture", {"window": win})      # 见下面「窗口被挡住」的坑
    return {n["key"]: n.get("value")
            for n in c.call("tree", {"window": win})["nodes"] if n.get("key")}

def act(key, action, text=None):
    params = {"window": win, "key": key, "action": action}
    if text is not None:
        params["text"] = text
    c.call("act", params)

# 填入中文，按翻译，等结果
act("source", "setValue", "我喜欢人工智能。今天天气很好。")
act("translate", "press")
deadline = time.time() + 60
while time.time() < deadline:
    time.sleep(0.4)
    value = state(refresh=True).get("result")
    if value:
        print("译文：", value)
        break

# 截图
png = c.call("capture", {"window": win})
open("/tmp/app.png", "wb").write(base64.b64decode(png["png"]))
```

跑：

```sh
go build -tags e2e -o /tmp/mygo-e2e .
MYGO_AUTOMATION=/tmp/mygo-e2e.sock /tmp/mygo-e2e &
python3 e2e/drive.py
```

---

## 坑

### 系统合成的鼠标键盘事件进不去原生 UI 窗口

用 AppleScript（`System Events` 的 `click at {x, y}`、`keystroke`）点 MyGo 的原生 UI 窗口，
**什么都不会发生**：窗口是 MyGo 自己画的，不走 AppKit 的控件事件通道。

```sh
osascript -e 'tell application "System Events" to click at {851, 246}'   # 无效
```

**用插件的 `act` / `input`**，它直接走 app 自己的事件通道。
`act` 还能按 key 找元素，比按坐标点稳。

### 窗口被挡住的时候不画帧，元素树停在旧的一帧

这个坑最费时间。症状是「翻译结果永远不回来」：

- app 确实改了状态、也确实调了 `Window.Update`
- 但 `tree` 报的还是**最后一帧**里的元素
- 窗口被别的窗口盖住时，macOS 不给它画帧 → 那一帧永远停在旧内容 → 树也停在旧内容

看起来像功能没实现，其实是没画。

**两个解法**：

1. **读 `tree` 之前先 `capture` 一次** —— `capture` 会强制渲染一帧，树就跟上了。窗口在不在前台都行。
2. **把 app 置前**，人眼也能看到：

   ```sh
   osascript -e 'tell application "System Events" to set frontmost of process "mygo-e2e" to true'
   ```

   驱动脚本里通常两个都做。

### `idle` 不等于「我的操作生效了」

`idle` 等的是**内容不再要求画帧**。一个一直转的菊花（`ui.Spinner`）会让它永远不返回，
所以要给 `timeout`：

```json
{"id": 8, "method": "idle", "params": {"window": 1, "timeout": 5000}}
```

而且**后一步依赖前一步的结果时**（比如先点一个字段再往里打字），中间要 `idle` 一下，
否则事件会在前一步的帧画出来之前就到。

### `tree` 里的元素只有辅助技术能读到的那些

布局用的行、列不出现在树里。角色有 `group`、`text`、`button`、`textField`、`popUpButton` 等。
`states` 有 `focusable`、`disabled`、`readOnly`、`multiline` 等。

**输入框的 `value` 是它的内容**，所以译文在不在、按钮是不是禁用，都能从树里读出来。

### `act` 找不到元素

`no_element` 有两种原因：key 写错了，或者**这一帧里没有这个元素**（比如按钮被隐藏了）。
`act` 只在**当前这一帧**的元素里找。

---

## 排查手册：结果不回来

按这个顺序查，别跳：

1. **请求发出去了吗？** 看模型那边有没有收到请求：
   ```sh
   curl -s http://localhost:11434/api/ps
   ```
   没收到 → app 没发出去，看 `doTranslate` 有没有被调用。
2. **请求回来了吗？** 在 app 里加临时日志，打印请求的起止：
   ```go
   log.Printf("request start %q", text)
   out, err := a.doTranslate(from, to, text)
   log.Printf("request done err=%v out=%q", err, out)
   ```
   `err` 有值 → 是 HTTP 层的问题（连不上、模型报错、超时）。
3. **状态更新了吗？** 再打一条，落在 `publish` 里面：
   ```go
   log.Printf("publish running")
   ```
   没打 → `publish` 没被调用，或者 goroutine 卡在 HTTP 上。
4. **帧画出来了吗？** `capture` 一次看图。**图里是对的但树里是旧的 → 就是遮挡那个坑。**

第 4 步最容易被忽略：状态对了、图也画出来了，只有树是旧的，就说明该 `capture` 一下再读树。

---

## 安全

**能打开这个 socket 的人，就能像用户一样操作这个 app**，还能在它的网页里跑脚本。

所以：

- **别把插件带进你发布的构建** —— 用构建标签关住，正式构建用 `mygo build`（它不带 inspector，
  本来就不含插件）。
- socket 放在**只有当前用户能写**的目录里，不要放共享目录。
