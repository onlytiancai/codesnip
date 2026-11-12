# 迁移前的原始实现（冻结）

这个目录里的文件是 **迁移前** `server/src/prompts/*.ts` 的原样拷贝，
从 git 里抽出来后不再改动：

```bash
git show HEAD:python/ai/035-tiptap-demo/server/src/prompts/<name>.ts \
  > server/tests/legacy/prompts/<name>.ts
```

存在的唯一目的：给 `server/tests/promptParity.test.ts` 当**对照**用 ——
提示词从硬编码搬到 JSON 模板之后，渲染结果必须和这里的 builder 输出
逐字节相同。

为什么不把期望值直接抄进测试文件：抄的时候会手抖。开发这个功能时就把
`"do not include it in the output"` 抄成了 `"... in your output"`，
结果「模板」和「测试里的期望值」一起错，测试反而是绿的 —— 绿得毫无意义。
冻一份真代码过来，才是真的对照。

**不要修改这个目录里的任何文件。**
