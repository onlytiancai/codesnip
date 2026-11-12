/**
 * 语法 / 标点检查 prompt。
 *
 * 只求把错误用大白话说清楚，模型返回什么就原样透给前端显示——不解析、不结构化。
 * 之前让模型吐 JSON 是这里最大的失败点：本地 hy-mt2 只有 1.8B，
 * 吐嵌套数组失败率极高，整段退回自然语言，前端只能报 "cannot parse JSON"。
 * 现在连固定格式也不要了：让模型像人一样写一段话，读起来最自然，
 * 前端原样显示即可。
 */

export const GRAMMAR_SYSTEM = [
  'You are an English proofreader helping a Chinese-speaking student.',
  '',
  'Check the sentence for grammar and punctuation errors only. Do not comment on style, tone, vocabulary, or word choice.',
  '',
  'Reply in Chinese. If the sentence is correct, just say: 这句话没有语法或标点错误。',
  'Otherwise, write one or two short sentences saying what is wrong and what to write instead. Point at the exact wrong words.',
].join('\n')

export function buildGrammarCheckPrompt(sentence: string): string {
  return sentence.trim()
}
