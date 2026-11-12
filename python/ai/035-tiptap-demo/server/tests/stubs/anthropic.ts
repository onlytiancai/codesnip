/**
 * 测试用的 `@anthropic-ai/sdk` 桩（见 tests/stubs/README 说明）。
 */

export class Anthropic {
  constructor(public readonly options: unknown) {
    throw new Error(
      '测试里用的是 @anthropic-ai/sdk 桩。' +
        '真 SDK 行为请在测试文件里用 vi.mock("@anthropic-ai/sdk", ...) 替换。',
    )
  }
}

export default Anthropic
