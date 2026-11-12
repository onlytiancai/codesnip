/**
 * 翻译的返回值类型与解析器。
 *
 * 提示词本身已经搬到 server/src/config/defaults.ts（可配置、可版本化）。
 */

import { cleanText } from './cleanText.js'

export interface TranslationResult {
  translation: string
}

/**
 * 解析模型响应为纯英文一句话。
 */
export function parseTranslation(raw: string): string {
  return cleanText(raw)
}
