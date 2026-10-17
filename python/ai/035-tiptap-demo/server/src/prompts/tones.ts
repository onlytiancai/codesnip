/**
 * 语气枚举与喂给模型的英文描述。
 *
 * 前端 `src/constants/tones.ts` 有一份同 id 的中文 label，两边 id 必须对齐。
 * 描述写得具体一点——只丢一个 "formal" 这种词给小模型，它理解成什么全靠猜。
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

export function isToneId(v: unknown): v is ToneId {
  return typeof v === 'string' && (TONE_IDS as readonly string[]).includes(v)
}

export const TONE_GUIDANCE: Record<ToneId, string> = {
  neutral:
    'keep it neutral and matter-of-fact — no strong colouring, no obvious personality',
  formal:
    'write formally — precise wording, no contractions, no slang, avoid first-person asides',
  casual:
    'write casually and conversationally — natural spoken rhythm, contractions are fine',
  professional:
    'write professionally — like a domain expert addressing peers: technical where relevant, concrete and measured',
  friendly:
    'write in a friendly, warm voice — approachable and encouraging, but still clear',
  concise:
    'write concisely — cut filler and redundant modifiers, keep every word load-bearing',
  detailed:
    'expand with detail — spell out the reasoning and add the specifics a reader would need, without padding',
  persuasive:
    'write persuasively — lead with the strongest point, use concrete language, build toward a convincing close',
}

/** 拼出喂给模型的一句 tone 指令。tone 非法或缺省时返回 undefined。 */
export function toneInstruction(tone: unknown): string | undefined {
  if (!isToneId(tone)) return undefined
  return `Write in a ${tone} tone: ${TONE_GUIDANCE[tone]}.`
}
