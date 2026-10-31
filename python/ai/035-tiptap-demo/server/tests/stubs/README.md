# 测试用的 SDK 桩

`ollama` / `openai` / `@anthropic-ai/sdk` 三个真 SDK **光 import 就要 ~180ms**
（openai ~84ms、ollama + anthropic ~93ms）。

而仓库里 13 个测试文件都会间接 import 到它们 —— 每个文件 ~180ms，
一轮测试就是 ~2.3s 的纯 import 开销，还是每个文件**重复**付一遍
（vitest 默认每个测试文件一份干净的模块表）。

真 SDK 在测试里**本来就没被真正执行过**：
`tests/llmClients.test.ts` 用 `vi.mock(..., factory)` 把三个包整个替换掉了，
其余测试压根不走 LLM 客户端。

所以 `vitest.config.ts` 里用 `resolve.alias` 把这三个包指向本目录下的桩，
import 成本变成 0。

**三个包必须是三个独立文件**：vitest 的 mock 注册表按「解析后的路径」做 key，
三个包 alias 到同一个文件的话，`vi.mock` 会互相覆盖
（症状：报 `No "Ollama" export is defined on the "@anthropic-ai/sdk" mock`）。

桩被真的调用时会抛错提醒你补 `vi.mock`，不会静默返回假数据。
