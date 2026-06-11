# RSS Reader CLI

基于 Textual TUI 的终端 RSS 阅读器。

## 功能

- 在终端浏览 RSS 订阅源
- 以 Markdown 格式阅读文章
- 标记文章已读/收藏
- SQLite 持久化存储
- 键盘导航
- 复制文章内容到剪贴板

## 安装

```bash
pip install -e .
```

## 使用

```bash
python -m rss_reader          # 默认日志级别 WARNING
python -m rss_reader -v       # INFO 级别
python -m rss_reader -vv      # DEBUG 级别
python -m rss_reader -q       # 静默模式，只显示 ERROR
```

## 快捷键

| 页面 | 按键 | 功能 |
|------|------|------|
| **全局** | `q` | 退出 |
| **订阅源列表** | `a` | 添加订阅源 |
| **订阅源列表** | `d` | 删除选中的订阅源 |
| **订阅源列表** | `j/k` | 上/下选择 |
| **订阅源列表** | `l` / `Enter` | 进入文章列表 |
| **订阅源列表** | `Ctrl+r` | 刷新所有订阅源 |
| **文章列表** | `h` | 返回订阅源列表 |
| **文章列表** | `j/k` | 上/下选择 |
| **文章列表** | `l` / `Enter` | 进入文章详情 |
| **文章详情** | `h` | 返回文章列表 |
| **文章详情** | `j/k` | 上/下滚动一行 |
| **文章详情** | `c` | 复制选中文字 |
| **文章详情** | `o` | 在浏览器打开 |
| **添加订阅源对话框** | `Enter` | 添加 |
| **添加订阅源对话框** | `Esc` | 取消 |

## 导航

```
[订阅源列表] --l/Enter--> [文章列表] --l/Enter--> [文章详情]
     ↑                      |
     └────── h (返回) ────┘
```

## 项目结构

```
src/rss_reader/
├── __main__.py         # 入口: python -m rss_reader
├── app.py              # 主程序和界面
├── models/
│   ├── feed.py         # Feed, Article 数据类
│   └── database.py     # SQLite 操作
└── services/
    ├── fetcher.py      # HTTP 请求
    ├── parser.py       # RSS 解析
    └── html2md.py      # HTML 转 Markdown
```

## 文章标记

- ~~`★` 黄色 - 已收藏~~
- ~~`●` 绿色 - 已读~~
- 无标记 - 未读

## 日志

日志写入 `rss_reader.log`，每天一个文件，保留最近 7 天。使用 `-v` 参数可在终端查看日志输出。