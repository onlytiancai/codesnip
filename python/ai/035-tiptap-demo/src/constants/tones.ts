/**
 * 语气枚举 —— 前端唯一的来源。
 *
 * 翻译下拉和「改写语气」二级菜单共用这里。
 * id 必须与后端 server/src/prompts/tones.ts 的 TONE_IDS 完全一致。
 */

export const TONES = [
  { id: 'neutral', label: '中性' },
  { id: 'formal', label: '正式' },
  { id: 'casual', label: '口语' },
  { id: 'professional', label: '专业' },
  { id: 'friendly', label: '友好' },
  { id: 'concise', label: '简洁' },
  { id: 'detailed', label: '详细' },
  { id: 'persuasive', label: '有说服力' },
] as const

export type ToneId = (typeof TONES)[number]['id']

export function isToneId(v: unknown): v is ToneId {
  return typeof v === 'string' && (TONES as readonly { id: string }[]).some((t) => t.id === v)
}

export function toneLabel(id: ToneId): string {
  return TONES.find((t) => t.id === id)?.label ?? id
}
