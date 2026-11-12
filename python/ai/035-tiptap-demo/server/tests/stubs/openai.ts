/**
 * 测试用的 `openai` SDK 桩（见 tests/stubs/README 说明）。
 */

export class OpenAI {
  constructor(public readonly options: unknown) {
    throw new Error(
      '测试里用的是 openai SDK 桩。' +
        '真 SDK 行为请在测试文件里用 vi.mock("openai", ...) 替换。',
    )
  }
}

export default OpenAI
