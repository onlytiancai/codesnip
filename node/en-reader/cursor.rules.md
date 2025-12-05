# Cursor（高亮/光标）规则说明

本文档描述 `index.html` 示例应用中关于“光标/高亮（cursor/highlight）”的设计规则、语义与边界行为，便于后续维护或改造成更复杂的阅读控制器。

## 术语定义

- token：由 `tokenizePreserve()` 规则分出来的最小单元，可能是单词、数字、连字符词、撇号内词，或是单个标点符号。
- index：token 在 `wordBlocks` 数组中的零基位置（0..N-1）。
- sentenceIndex：在分析阶段分配给 token 的句子编号。句子编号按遇到句子分隔符（例如 `.` `!` `?` `，` `。` `;` 等）递增。
- highlight：布尔值，指示某个 token 是否被高亮。UI 中用 `.hl` 类显示高亮样式。

## 高亮控制函数

- highlightIndex(i)
  - 语义：仅将索引为 i 的单个 token 置为高亮，其他 token 取消高亮。
  - 边界行为：若 i < 0 或 i >= wordBlocks.length，则相当于清除所有高亮。

- highlightRange(startIdx, endIdx)
  - 语义：将索引在 [startIdx, endIdx] 范围内的 token 都置为高亮，范围外的 token 取消高亮。
  - 特殊用法：代码中使用 `highlightRange(-1, -2)` 来清除全部高亮（因为没有 i 满足 i >= -1 && i <= -2）。
  - 边界行为：当 startIdx > endIdx 或两端超出范围时，函数仍按简单比较逻辑处理，但常用做法是用特定负值来清空高亮。

## 与朗读同步的规则

- 逐词朗读（speak）
  - 在每个 utterance 的 onstart 回调中调用 `highlightIndex(i)`，将正在朗读的 token 高亮。
  - 在 onend 或 onerror 回调中调用 `highlightIndex(-1)` 来清除高亮。
  - 只对包含字母或数字的 token 触发朗读与高亮（纯标点跳过）。

- 逐句朗读（speakSentences）
  - 在句子 utterance 的 onstart 中调用 `highlightRange(start, end)` 来高亮整句。
  - 在 onend 或 onerror 中调用 `highlightRange(-1, -2)` 来清除高亮。
  - 句子边界由分析阶段分配的 `sentenceIndex` 决定，默认的分句字符集包含 `.,!?，。！；;`。

- 点击单词（speakWord）
  - 立即停止当前朗读，创建单个 utterance 并在 onstart 中 `highlightIndex(i)`，在 onend 中 `highlightIndex(-1)`。

- 停止（stop）
  - 设定 `stopRequested = true`，调用 `speechSynthesis.cancel()`，并调用 `highlightIndex(-1)` 以清空所有高亮。

## 光标/高亮的显示约定

- 高亮仅为视觉提示，不携带播放进度精度信息（即无子词或发音片段的微粒级高亮）。
- 高亮样式使用 `.hl` 类（黄色渐变背景），且在单词上应用（`.word-top` 元素）。
- 当高亮范围跨越多个 token（逐句朗读），所有被包含的 token 同时应用 `.hl`。

## 可配置项与扩展建议

1. 支持子词高亮（phones/phoneme alignment）
   - 若后端或库能提供字词到音素（phones）级别的时间轴，可以为每个音素创建子高亮并在 utterance 的 boundary 事件中逐步更新高亮。

2. 可视化播放进度
   - 在每个 token 上维护播放时长和当前播放偏移，显示一个进度条或微小动画，使用户能看到更细粒度的进度。

3. 自定义分句规则
   - 目前使用固定标点分句。可增加缩写识别（`Mr.`, `Dr.`）和括号/引号处理以减少误分句。

4. 键盘导航与快捷键
   - 建议实现：Left/Right 移动 word index，Up/Down 跳到上一/下一句，Space 播放/暂停，Esc 停止并清除高亮。

## 边界与异常场景

- 空文本：分析或朗读时直接返回，不影响现有状态。
- 连续分隔符（例如 `...` 或 `!!!`）：每个分隔符都会触发 sentenceIndex 的递增；可以考虑合并连续的分隔符为一个句子终结符。
- 非英文字符或表情：被分词为独立 token，若不包含字母/数字则不会被朗读并不会触发高亮。
- 本地缓存 IPA 失败或格式不一致：fetchIPA 会保守地退回到离线映射或 null，UI 用 '—' 显示缺失的音标。

## 测试建议

- 用包含缩写、数字、撇号、连字符与多种标点的文本测试分析和分句正确性。
- 在不同浏览器（Chrome/Edge/Safari）上测试朗读行为和语音可用性。
- 测试 stop/重启 的鲁棒性：快速点击停止与播放按钮需确保高亮与音频状态同步。

---

以上规则基于 `index.html` 中的实现细节总结，便于后续将光标逻辑提取为模块或替换为更高级的播放同步器。
