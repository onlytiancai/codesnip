# CET4 / 高中 / 初中 词汇查看器

合并去重后的 5011 词（`vocab_all.json` 12 MB JSONL + `vocab_all.idx` 140 KB 索引），
提供两套访问入口：

- **macOS GUI 应用** `VocabViewer`（`main.swift` + swiftc）
- **命令行** `scripts/cet4.py`（Python 3，标准库）

数据生成器：`scripts/build-vocab.py`（一次性把 `CET4_2.json` /
`GaoZhongluan_2.json` / `ChuZhongluan_2.json` 按 wordHead 去重合并）。

---

## 编译 macOS 应用

```bash
swiftc \
  -sdk "$(xcrun --sdk macosx --show-sdk-path)" \
  -framework AppKit \
  main.swift \
  -o VocabViewer
```

## 运行

```bash
./VocabViewer                          # 用 Bundle 同目录或 cwd 的 vocab_all.idx
./VocabViewer --data /path/to/dir      # 自定义数据目录
```

如果 `vocab_all.idx` 不存在，会提示先跑构建脚本（见下）。

### 操作

- 启动后随机显示一个词
- 搜索框输入 `abandon` 回车 — 显示 `abandon` 详情
- `⌘N` — 下一个随机词
- `⌘F` — 聚焦搜索框
- `⌘C` / `⌘V` / `⌘A` / `⌘Z` — 在文本视图里复制/粘贴/全选/撤销（依赖 Edit 菜单）
- `⌘Q` — 退出

### 找不到时的候选

输入 `abandons` 会渲染：

```
未找到 "abandons"。你可能想找:
  - abandon          (0.95)
  - abundant         (0.62)
  - anyone           (0.50)
  ...
```

候选算法与 `scripts/cet4.py suggest_similar` 等价（Levenshtein +
后缀剥离 + 前缀延伸 tie-breaker）。

---

## 命令行

```bash
python3 scripts/cet4.py                 # 随机
python3 scripts/cet4.py abandon         # 指定
python3 scripts/cet4.py abandons        # 找不到 → 候选
```

## 重新生成数据

```bash
python3 scripts/build-vocab.py
```

把 `CET4_2.json` / `GaoZhongluan_2.json` / `ChuZhongluan_2.json` 合并去重后
写入 `vocab_all.json`（JSONL 格式），索引 `vocab_all.idx` 会在 `cet4.py` /
`VocabViewer` 首次运行时按 mtime 校验自动重建。
