# Design Tokens · 完整配色系统

「茶韵 · 宋雅」的设计语言分解。每个 token 都对应一个明确的使用场景。

## 1. 背景色阶（纸）

低饱和暖米色，模拟宣纸的层次感。

| Token | 值 | 用途 | 不要用于 |
|---|---|---|---|
| `--paper` | `#f5f1ea` | 页面主背景 | 面板（会显灰） |
| `--paper-2` | `#faf7f1` | 输入框、播放器等"凹陷"区域 | 卡片（会和 paper 重复） |
| `--card` | `#fdfbf6` | 面板、卡片、抽屉 | 页面背景（会显白刺眼） |
| `--card-2` | `#f8f4ec` | 抽屉头部、按钮悬浮态 | 大面积 |
| `--line` | `#e8e1d3` | 边线、分割线 | 强对比边框 |
| `--line-2` | `#d9d0bd` | 深边线、次要按钮边 | 主边线（太重） |

**为什么要这么多层？** 模拟宣纸在不同光线下的层次 —— 折叠处更深，迎光处更浅。

## 2. 文字色阶（墨）

| Token | 值 | 用途 |
|---|---|---|
| `--ink-1` | `#2e2a24` | 主文字、标题、按钮文字 |
| `--ink-2` | `#5b5247` | 次文字、面板 h2、按钮文字（次要按钮） |
| `--ink-3` | `#8a8273` | 辅文字、提示语、meta 信息 |
| `--ink-4` | `#b3aa9b` | 灰墨、占位符、disabled 文字 |

**墨阶规律**：每个色阶亮度差 25-30 点，色相差 5-10 度，模拟墨在不同纸张上洇开的效果。

## 3. 点彩色（关键！）

点彩是整套配色的"魂"，但 **绝不能大面积使用**。每种点彩都有明确的使用范围。

### 3.1 青瓷绿（主点彩）

取自宋代汝窑、龙泉青瓷的颜色。

| Token | 值 | 用途 |
|---|---|---|
| `--celadon` | `#6f7e6a` | 主操作按钮、选中态、单选框、链接 hover |
| `--celadon-2` | `#5b6a56` | 主按钮悬浮态、强调标题 |
| `--celadon-bg` | `#e9eee5` | 输入框聚焦晕染、徽章背景、按钮次态 |

**禁忌**：不要把整块区域涂成青瓷绿（哪怕淡色），会显得"闷"。

### 3.2 赭石（次点彩）

取自古画山水里的暖色勾线。

| Token | 值 | 用途 |
|---|---|---|
| `--ochre` | `#a07a5b` | 次强调、链接默认色、面包屑 |
| `--ochre-bg` | `#f1e9dd` | 极少用，预留 |

### 3.3 朱砂（特殊点彩）

| Token | 值 | 用途 |
|---|---|---|
| `--cinnabar` | `#a8412e` | 印章 logo、特殊状态（如"重要"标记）、错误边框点缀 |

**严格控制用量**：一个页面最多 1-2 个朱砂元素（一般是 logo）。

### 3.4 语义色

| 语义 | Token | 值 | 配背景 |
|---|---|---|---|
| 错误 | `--clay` | `#b56b5a` | `--clay-bg` `#f4e3dd` |
| 成功 | `--moss` | `#5e7d65` | `--moss-bg` `#e3ebe2` |
| 信息 | （复用 `--celadon`） | `#6f7e6a` | `--celadon-bg` |

**注意**：信息色复用青瓷绿，不引入新的蓝色，保持"清雅"统一感。

## 4. 阴影

| Token | 值 | 用途 |
|---|---|---|
| `--shadow-sm` | `0 1px 2px rgba(76, 60, 30, 0.04)` | 卡片悬浮 |
| `--shadow` | `0 1px 2px rgba(76, 60, 30, 0.04), 0 8px 28px rgba(76, 60, 30, 0.05)` | 面板默认 |
| `--shadow-lg` | `0 4px 12px rgba(76, 60, 30, 0.06), 0 20px 50px rgba(76, 60, 30, 0.08)` | 抽屉、模态框 |

**关键**：所有阴影的色相都是暖色（`rgba(76, 60, 30, ...)`），不是黑色。冷色阴影会破坏"宣纸"质感。

## 5. 圆角

| Token | 值 | 用途 |
|---|---|---|
| `--radius` | `12px` | 面板、卡片 |
| `--radius-sm` | `8px` | 按钮、输入框、小卡片 |
| `--radius-lg` | `16px` | 模态框、特殊大卡片 |

**节奏感**：页面里至少要有两种圆角交替使用，避免视觉单调。

## 6. 字体

| Token | 值 | 用途 |
|---|---|---|
| `--font-sans` | `-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif` | 正文、按钮、表格 |
| `--font-serif` | `"Noto Serif SC", "Songti SC", "STSong", serif` | 页面大标题、面板小标题、徽章 logo |
| `--font-mono` | `"SF Mono", Menlo, Consolas, monospace` | 代码、文件名、ID、数字 |

**衬线字加载**：

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@400;500;600;700&display=swap" rel="stylesheet" />
```

## 7. 完整 token 块（直接复制）

```css
:root {
  /* —— 纸 · 墨 —— */
  --paper:        #f5f1ea;
  --paper-2:      #faf7f1;
  --card:         #fdfbf6;
  --card-2:       #f8f4ec;
  --line:         #e8e1d3;
  --line-2:       #d9d0bd;

  /* —— 墨阶 —— */
  --ink-1:        #2e2a24;
  --ink-2:        #5b5247;
  --ink-3:        #8a8273;
  --ink-4:        #b3aa9b;

  /* —— 点彩 —— */
  --celadon:      #6f7e6a;
  --celadon-2:    #5b6a56;
  --celadon-bg:   #e9eee5;
  --ochre:        #a07a5b;
  --ochre-bg:     #f1e9dd;
  --cinnabar:     #a8412e;
  --clay:         #b56b5a;
  --clay-bg:      #f4e3dd;
  --moss:         #5e7d65;
  --moss-bg:      #e3ebe2;

  /* —— 阴影 —— */
  --shadow-sm:    0 1px 2px rgba(76, 60, 30, 0.04);
  --shadow:       0 1px 2px rgba(76, 60, 30, 0.04), 0 8px 28px rgba(76, 60, 30, 0.05);
  --shadow-lg:    0 4px 12px rgba(76, 60, 30, 0.06), 0 20px 50px rgba(76, 60, 30, 0.08);

  /* —— 圆角 —— */
  --radius:       12px;
  --radius-sm:    8px;
  --radius-lg:    16px;

  /* —— 字体 —— */
  --font-sans:    -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                  "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
  --font-serif:   "Noto Serif SC", "Songti SC", "STSong", serif;
  --font-mono:    "SF Mono", Menlo, Consolas, monospace;
}
```

## 8. 调色逻辑（设计 rationale）

为什么是这些颜色？因为它们在 HSL 空间里都共享几个特征：

- **色相（H）**：全部在 30°-110° 范围（暖色相，橙-黄-绿），没有任何冷蓝紫
- **饱和度（S）**：全部低于 25%，绝不"艳"
- **明度（L）**：背景 95%+，文字 15-30%，点彩 35-45%
- **色温差**：暖色和冷色靠"墨阶"过渡，不是靠饱和度对比

如果用户要求做一个 **暗色版本**，**不要简单 invert 颜色**！需要重新设计：
- 背景换成深墨色（如 `#1a1714`）
- 文字换成米色系（如 `#e8e1d3`）
- 点彩保持青瓷绿但调亮 20%（如 `#8a9a85`）
- 阴影换成深色 + 微光晕
