# Wawa English Speaker

这是一个使用 Vue 3 和 浏览器 Speech Synthesis API 的示例应用，演示如何：

- 将输入文本进行分词（tokenize），保留原文标点与撇号。
- 为每个单词尝试获取音标（IPA），仅使用本地 CSV 文件 `config/offlineIPA.csv`（不再依赖外部在线词典，也不使用 localStorage 缓存）。
- 支持逐词朗读、逐句朗读，并在朗读时高亮当前单词或句子。
- 可调节语速（rate）与音调（pitch）。
