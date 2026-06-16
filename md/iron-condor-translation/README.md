# 铁鹰式期权学习测试 / Iron Condor Quiz

基于 [`iron-condor-bilingual.md`](./iron-condor-bilingual.md) 的章节学习测试题库。

- **45 道题** 覆盖全部 18 个 H2 章节（单选 36 + 多选 9）
- **按章节分组** 作答，顶部总进度条
- **提交后** 显示总体 + 每章节正确率，错题含详细解析
- 纯静态 HTML / Vue 3 / marked / DOMPurify（CDN）
- 明亮色彩方案，明暗友好的中文界面

## 启动方式

由于浏览器对 `file://` 的 fetch 限制，需要通过 HTTP 访问：

```bash
# 进入目录
cd iron-condor-translation

# 启动本地服务器（任选其一）
pnpm dlx http-server . -p 8765 -c-1
# 或
python3 -m http.server 8765

# 浏览器打开
open http://127.0.0.1:8765/index.html
```

## 文件结构

```
iron-condor-translation/
├── iron-condor-bilingual.md    # 原文（不动）
├── images/                     # 配图（不动）
├── quiz.md                     # 题目源（新增，可自由编辑）
├── index.html                  # 入口
├── parser.js                   # 题目 markdown 解析器
├── app.js                      # Vue 3 应用
├── style.css                   # 样式
└── README.md                   # 本文件
```

## 题目 markdown 自定义语法

详见 `parser.js`。核心结构：

```markdown
<!-- section: <key> -->        <!-- 章节 key（可选） -->

## 章节标题 / Section Title    <!-- 章节 -->

::: quiz <id> <single|multiple>  <!-- 题块开始 -->
id: q-xxx
section: <key>
difficulty: easy|medium|hard
tags: tag1, tag2
question: |
  题干（markdown，`|` 续行）
options:
  - { key: A, text: "选项 A" }
  - { key: B, text: "选项 B" }
  - { key: C, text: "选项 C" }
  - { key: D, text: "选项 D" }
answer: A                  # 单选一个 key；多选用 "A, C"
explanation: |
  解析（markdown）
points: 1                   # 可选，默认单选 1 / 多选 2
:::                          <!-- 题块结束 -->
```

## 出题人流程

1. 编辑 `quiz.md`，按上述语法新增 / 修改题目
2. 浏览器刷新 `http://127.0.0.1:8765/index.html` 即可看到最新内容
3. 无需重新构建、无需重启服务器

## 字段速查

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | ✓ | 题号（题块首行已声明） |
| `section` | ✓ | 章节 key（与 `<!-- section: -->` 对应） |
| `type` | ✓ | `single`（单选） / `multiple`（多选） |
| `difficulty` | ✗ | `easy` / `medium` / `hard` |
| `tags` | ✗ | 逗号分隔 |
| `question` | ✓ | 题干 markdown |
| `options` | ✓ | `- { key, text }` |
| `answer` | ✓ | 单选一个 key；多选 `A, C` |
| `explanation` | ✓ | 解析 markdown |
| `points` | ✗ | 默认单选 1 / 多选 2 |

## 评分规则

- 单选：选对得 1 分，选错 0 分
- 多选：完全正确得 2 分；部分对（漏选但未错选）得 1 分；含错选 0 分
- 未作答的题不计入得分

## 技术栈

- **Vue 3.4.27** — `https://cdn.staticfile.org/vue/3.4.27/vue.global.prod.js`
- **marked 12.0.2** — `https://cdn.staticfile.org/marked/12.0.2/marked.min.js`
- **DOMPurify 3.1.6** — `https://cdn.staticfile.org/dompurify/3.1.6/purify.min.js`

无 npm 依赖、无构建步骤，纯静态。
