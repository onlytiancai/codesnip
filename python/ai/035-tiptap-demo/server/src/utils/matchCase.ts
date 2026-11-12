/**
 * 把 target 的大小写调整为与 source 一致：
 * - "GOOD" + "great"  → "GREAT"  （全大写 → 全大写）
 * - "Good" + "great"  → "Great"  （首字母大写 → 首字母大写）
 * - "good" + "Great"  → "great"  （全小写 → 全小写）
 *
 * 单字符的 "I" / "i" 走"首字母"分支，符合常规。
 */
export function matchCase(source: string, target: string): string {
  if (!target) return target
  if (!source) return target

  // 全部字母都大写（"GOOD"、"USA"），且确实含字母字符
  // 单字符（"I"）不算全大写——走"首字母大写"分支更符合英文习惯
  const hasLetters = /[A-Za-z]/.test(source)
  if (
    source.length > 1 &&
    hasLetters &&
    source === source.toUpperCase() &&
    source !== source.toLowerCase()
  ) {
    return target.toUpperCase()
  }

  // 首字母大写（"Good"、"Apple"）
  const first = source[0]
  if (first && first === first.toUpperCase() && first !== first.toLowerCase()) {
    return (target[0] ?? '').toUpperCase() + target.slice(1)
  }

  // 其余（首字母小写 / 无大小写概念）→ 保持 target 原样
  return target
}