import { vi } from 'vitest'
import { createApp } from '../src/app.js'
import type { AiService } from '../src/services/ollama.js'
import type { TranslateService } from '../src/services/translate.js'

/**
 * 路由层测试的公共脚手架：注入 mock service，直接打到真实路由。
 */

/** 默认各能力返回值；传 overrides 覆盖单个方法。 */
export function buildAiService(overrides?: Partial<AiService>): AiService {
  return {
    rewriteSelection: vi.fn(async () => ['great', 'promising', 'excellent']),
    rewriteSentence: vi.fn(async () => 'Education is of great importance to children.'),
    grammarCheck: vi.fn(async () => '这句话没有语法或标点错误。'),
    rewriteInstruction: vi.fn(async () => 'Rewritten text.'),
    ...overrides,
  }
}

export function buildTranslateService(
  overrides?: Partial<TranslateService>,
): TranslateService {
  return {
    translate: vi.fn(async () => ({ translation: 'I went to the park yesterday.' })),
    ...overrides,
  }
}

export function buildApp(
  ai?: AiService,
  translate?: TranslateService,
): ReturnType<typeof createApp> {
  return createApp({
    aiService: ai ?? buildAiService(),
    translateService: translate ?? buildTranslateService(),
  })
}
