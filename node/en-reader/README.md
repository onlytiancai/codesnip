# Speech + IPA Demo（Vue3）

这是一个使用 Vue 3 和 浏览器 Speech Synthesis API 的示例应用，演示如何：

- 将输入文本进行分词（tokenize），保留原文标点与撇号。
- 为每个单词尝试获取音标（IPA），优先从在线词典拉取，失败时使用内置离线映射并缓存到 localStorage。
- 支持逐词朗读、逐句朗读，并在朗读时高亮当前单词或句子。
- 可调节语速（rate）与音调（pitch）。

该项目仅依赖浏览器环境（无需后端），并引入了 Vue 3 的 UMD 构建与 Tailwind 来做样式。

## 主要文件

- `index.html` - 应用的全部代码（HTML / CSS / JavaScript）。

## 功能说明

1. 分词规则（tokenizePreserve）
   - 使用正则 `/[A-Za-z0-9\u2019'-]+|[^\s]/gu`：
     - 将连续的字母、数字、ASCII 单引号（'）、Unicode 右单引号（’）和连字符视为一个 token（例如：`1950s`, `Shannon’s`, `mother-in-law`）。
     - 其余非空白字符作为单个 token（保留标点）。

2. 音标获取（fetchIPA）
   - 先从 localStorage 缓存读取。
   - 尝试请求 `https://api.dictionaryapi.dev/api/v2/entries/en/{word}` 获取 phonetics.text。
   - 若请求失败或未命中，则退回到内置的 `offlineIPA` 映射。
   - 请求结果会写回 localStorage 以便下次使用。

3. 朗读逻辑
   - 逐词朗读（speak）：对每个包含字母或数字的 token 生成单独的 SpeechSynthesisUtterance，并在 onstart 时高亮对应单词，onend 时取消高亮，逐个等待完成。
   - 逐句朗读（speakSentences）：先根据分词结果构建 sentenceIndex（以 `.[,!?，。！；;]` 等字符分句），将相同句子索引的 token 集合在一起合并为一句，用一个 utterance 朗读并高亮整句。
   - 点击单词可单独朗读该词（speakWord）。
   - 有 `stop` 按钮可取消当前朗读并清除高亮。

4. UI 控件
   - 文本输入框（textarea）用于输入/编辑要朗读的文本。
   - 按钮：分析（分析并显示单词与音标）、逐词朗读、逐句朗读、停止。
   - 语速 & 音调滑块（rate, pitch）。

## 使用方法

1. 直接在支持 Web Speech API 的现代浏览器中打开 `index.html`。
2. 在文本框输入或修改英文文本，点击 “分析” 以显示分词结果与音标（若有）。
3. 使用 “逐词朗读” 或 “逐句朗读” 按钮听读并观察高亮。
4. 点击某个单词可以单独朗读该词。

## 注意事项 & 限制

- Speech Synthesis 在不同浏览器/平台上的表现不同，可能没有可用语音或音色有所差异。建议使用 Chrome / Edge / Safari 的最新版本进行测试。
- 依赖第三方词典 API（dictionaryapi.dev），请求频率受限且可能跨域失败；已提供 `offlineIPA` 作离线回退。
- 分词规则是按英文场景设计的，非字母的 token（纯标点）不会被朗读。
- localStorage 用于缓存 IPA，若需清理请在浏览器开发者工具中删除对应 `ipa_cache_v1` 键。

## 可扩展方向（建议）

- 提供语言选择（en-US / en-GB 等）和更丰富的声音列表。
- 改进 sentence 分割逻辑，支持缩写（e.g. "Mr.") 不被误断句。
- 将音标服务替换为更稳定的后端或批量查询接口以减少延迟。
- UI/UX：为高亮动画、播放进度等添加更细粒度的控制。

---

项目由 Vue 3（UMD）和原生 Web Speech API 实现，适合学习与演示用途。欢迎基于此示例扩展更多功能。
