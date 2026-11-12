/**
 * 语气枚举 —— 前端 id 的唯一来源。
 *
 * id 必须与后端 server/src/config/defaults.ts 的 TONE_IDS 完全一致。
 * label 默认取这里；实际生效的 label 从后端拉（每个 provider 可以改），
 * 拉不到就用这份内置的兜底。
 */

export const TONE_IDS = [
  'neutral',
  'formal',
  'casual',
  'professional',
  'friendly',
  'concise',
  'detailed',
  'persuasive',
] as const

export type ToneId = (typeof TONE_IDS)[number]

export interface ToneOption {
  id: ToneId
  label: string
}

export const TONES: ToneOption[] = [
  { id: 'neutral', label: '中性' },
  { id: 'formal', label: '正式' },
  { id: 'casual', label: '口语' },
  { id: 'professional', label: '专业' },
  { id: 'friendly', label: '友好' },
  { id: 'concise', label: '简洁' },
  { id: 'detailed', label: '详细' },
  { id: 'persuasive', label: '有说服力' },
]

export function isToneId(v: unknown): v is ToneId {
  return typeof v === 'string' && (TONE_IDS as readonly string[]).includes(v)
}

export function toneLabel(id: ToneId): string {
  return TONES.find((t) => t.id === id)?.label ?? id
}
