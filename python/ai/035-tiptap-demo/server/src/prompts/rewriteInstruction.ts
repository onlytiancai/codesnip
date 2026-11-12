/**
 * 自定义指令改写的解析器。
 *
 * 提示词本身已经搬到 server/src/config/defaults.ts（可配置、可版本化）。
 */

import { cleanText } from './cleanText.js'

export function parseInstructionResult(raw: string): string {
  return cleanText(raw)
}
