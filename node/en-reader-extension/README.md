# English Reader - 沉浸式翻译扩展

一个 Chrome 浏览器扩展，提供沉浸式英译中翻译功能，支持双语对照显示和语音朗读。

## 目录结构

```
en-reader-extension/
├── extension/                    # Chrome 扩展 (Manifest V3)
│   ├── manifest.json            # 扩展清单
│   ├── background.js            # Service Worker - 处理 API 调用
│   ├── content-script.js        # DOM 操作、HTML 清理、翻译注入
│   ├── selector-overlay.js      # DevTools 风格元素选择器
│   ├── styles.css               # 翻译样式和 TTS 按钮样式
│   ├── popup/                   # 扩展弹窗 UI
│   └── tests/                   # Jest 单元测试
│
└── server/                       # Express 后端服务
    ├── index.js                  # 主服务器 (端口 3000)
    ├── routes/
    │   ├── translate.js          # POST /api/translate 翻译接口
    │   └── tts.js               # POST /api/tts 语音合成接口
    ├── services/
    │   ├── translateService.js   # LLM 翻译服务（带缓存）
    │   └── ttsService.js        # Kokoro TTS 本地语音合成
    └── config/
        └── prompts.js            # 翻译提示词配置
```

## 功能特性

- **元素选择器**: DevTools 风格覆盖层，可精确选择内容容器
- **智能 HTML 清理**: 移除脚本、样式、广告、导航等无关元素
- **上下文感知翻译**: 将整个容器发送给 LLM，保证译文连贯
- **双语对照显示**: 译文显示在原文下方
- **语音朗读**: 点击 🔊 按钮朗读翻译内容

## 安装配置

### 1. 安装依赖

```bash
# 安装服务器依赖
cd server
pnpm install

# 安装扩展测试依赖（如需运行测试）
cd ../extension
pnpm install
```

### 2. 配置 API 密钥

编辑 `server/.env` 文件：

```env
ANTHROPIC_BASE_URL=https://api.minimaxi.com/anthropic
ANTHROPIC_API_KEY=your-api-key
ANTHROPIC_MODEL=MiniMax-M2.7
```

### 3. 加载 Chrome 扩展

1. 打开 Chrome，进入 `chrome://extensions/`
2. 启用右上角"开发者模式"
3. 点击"加载已解压的扩展程序"
4. 选择项目中的 `extension/` 文件夹

### 4. 启动服务器

```bash
cd server
pnpm start
```

服务器运行在 http://localhost:3000

## 使用方法

1. 点击 Chrome 工具栏中的扩展图标
2. 从下拉框选择语音（默认 af_bella）
3. 点击"Select Content"进入选择模式
4. 点击要翻译的文章区域
5. 译文会显示在每个段落下方
6. 点击 🔊 按钮听取朗读

## API 接口

### POST /api/translate

翻译 HTML 内容。

**请求：**
```json
{
  "html": "<p>Hello world</p>",
  "elements": [{"index": 0, "tag": "p", "text": "Hello world"}]
}
```

**响应：**
```json
{
  "translations": [{"index": 0, "translation": "你好，世界"}]
}
```

### POST /api/tts

语音合成。

**请求：**
```json
{
  "text": "你好",
  "voice": "af_bella"
}
```

**响应：** 音频文件 (audio/wav)

### GET /api/tts/voices

获取可用语音列表。

**响应：**
```json
{
  "voices": ["af_heart", "af_alloy", "af_bella", ...]
}
```

## 运行单元测试

### 服务器测试（4 个测试）

```bash
cd server
pnpm test
```

输出示例：
```
PASS tests/translateService.test.js
  translateService
    ✓ successfully returns translation results
    ✓ caches identical translations
    ✓ handles empty translation from LLM
    ✓ handles index out of bounds gracefully
```

### 扩展测试（22 个测试）

```bash
cd extension
pnpm test
```

输出示例：
```
PASS tests/content-script.test.js
  content-script
    cleanHTML
      ✓ removes script, style, noscript, iframe
      ✓ removes nav, header, footer, aside
      ...
    extractTranslatableElements
      ✓ extracts single p tag
      ✓ extracts multiple consecutive p tags
      ...
    injectTranslations
      ✓ inserts translation after p tag
      ✓ prevents duplicate injection
      ...
```

### 同时运行所有测试

```bash
# 服务器测试
cd server && pnpm test

# 扩展测试
cd ../extension && pnpm test
```

## TTS 语音服务

本项目使用本地 Kokoro TTS 模型进行语音合成：

- **模型路径**: `/Volumes/data/coscdn/onnx-community/Kokoro-82M-v1.0-ONNX`
- **模型类型**: ONNX 量化模型 (q8)
- **npm 包**: `kokoro-js`

可用语音包括：
- 美式女声: af_heart, af_alloy, af_bella, af_sarah, af_sky 等
- 美式男声: am_adam, am_eric, am_puck 等
- 英式女声: bf_alice, bf_emma 等
- 英式男声: bm_daniel, bm_george 等

## 核心函数说明

### content-script.js

- `cleanHTML(element)` - 清理不需要翻译的元素和属性
- `extractTranslatableElements(container)` - 提取 p, h1-h6, li, blockquote
- `injectTranslations(elements, translations)` - 将翻译注入 DOM
- `addTTSButtons(elements)` - 在翻译后添加播放按钮

### HTML 清理规则

移除以下元素：
- `script`, `style`, `noscript`, `iframe`
- `nav`, `header`, `footer`, `aside`
- 带有 `role="navigation"`, `role="banner"`, `role="complementary"` 的元素
- `.ad`, `.advertisement`, `.sidebar`, `.comment` 类名的元素
- `svg`, `img`, `video`, `audio`
- 事件处理器 (`on*` 属性)
- `data-*` 属性
