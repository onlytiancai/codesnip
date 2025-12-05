# Speech + IPA Demo（Vue3）

这是一个使用 Vue 3 和 浏览器 Speech Synthesis API 的示例应用，演示如何：

- 将输入文本进行分词（tokenize），保留原文标点与撇号。
- 为每个单词尝试获取音标（IPA），优先从在线词典拉取，失败时使用内置离线映射并缓存到 localStorage。
- 支持逐词朗读、逐句朗读，并在朗读时高亮当前单词或句子。
- 可调节语速（rate）与音调（pitch）。

该项目仅依赖浏览器环境（无需后端），并引入了 Vue 3 的 UMD 构建与 Tailwind 来做样式。