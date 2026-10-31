/**
 * 测试用的 `ollama` SDK 桩（见 tests/stubs/README 说明）。
 *
 * 真 SDK 光 import 就要 ~180ms，而 13 个测试文件都会间接 import 到它。
 * 真 SDK 在测试里本来就被 `vi.mock('ollama', factory)` 整个替换掉了
 * （见 tests/llmClients.test.ts），所以这里只需要一个能 import 的空壳。
 *
 * 真被调用时抛错说清楚 —— 比静默返回假数据好。
 */

export class Ollama {
  constructor(public readonly options: unknown) {
    throw new Error(
      '测试里用的是 ollama SDK 桩。' +
        '真 SDK 行为请在测试文件里用 vi.mock("ollama", ...) 替换。',
    )
  }
}

export default Ollama
