/**
 * 自定义指令改写 prompt。
 *
 * 用户给一段任意内容 + 一句自然语言指令（"Make it more academic." /
 * "Simplify the sentence." / "Rewrite it in a formal tone."），
 * 模型按指令返回改写结果。
 *
 * 指令原样进 user message——不要去解析它，小模型理解自然语言指令比
 * 我们写规则可靠得多。上下文作为参考附在后面。
 */

import { cleanText } from './cleanText.js'

export const INSTRUCTION_SYSTEM = [
  'You are an English writing editor. You rewrite the given text according to the user\'s instruction.',
  'Output ONLY the rewritten text — no quotes, no labels, no explanation, no restating the instruction.',
  'If the instruction asks you to preserve the original, keep it unchanged. Otherwise return the improved version.',
].join(' ')

export function buildInstructionPrompt(
  text: string,
  instruction: string,
  context?: string,
): string {
  const ctx = context?.trim()
  return [
    `Instruction: ${instruction.trim()}`,
    `Text to rewrite: ${text}`,
    ctx ? `Surrounding context (for coherence; do not include it in the output): ${ctx}` : '',
    'Output:',
  ]
    .filter(Boolean)
    .join('\n')
}

export function parseInstructionResult(raw: string): string {
  return cleanText(raw)
}
