/**
 * 「日桶键」和「月区间」的计算。
 *
 * day 用 INTEGER 的 YYYYMMDD（如 20261031）—— 不用 TEXT，因为 SQLite 没有
 * 真正的 DATE 类型，而这一列只做等值和范围比较，INTEGER 比较最干净。
 *
 * **一律用服务端本地时区**（getFullYear / getMonth / getDate）——
 * 不用 UTC，否则东八区晚上 8 点之后记到的是「明天」。
 */

/** 今天的日桶键，如 20261031。 */
export function dayNumber(date: Date = new Date()): number {
  const y = date.getFullYear()
  const m = date.getMonth() + 1
  const d = date.getDate()
  return y * 10000 + m * 100 + d
}

/**
 * 当月的区间：[起始日, 结束日) —— 结束日是「下月 1 号」，开区间。
 *
 * 用开区间是为了跨年不用特判：11 月 → [20261101, 20261201)，12 月 → [20261201, 20270101)。
 */
export function monthRange(date: Date = new Date()): { from: number; to: number } {
  const y = date.getFullYear()
  const m = date.getMonth() + 1
  const from = y * 10000 + m * 100 + 1
  const nextYear = m === 12 ? y + 1 : y
  const nextMonth = m === 12 ? 1 : m + 1
  const to = nextYear * 10000 + nextMonth * 100 + 1
  return { from, to }
}

/** 20261031 → '2026-10-31'（只用来显示）。 */
export function formatDay(day: number): string {
  const y = Math.floor(day / 10000)
  const m = Math.floor((day % 10000) / 100)
  const d = day % 100
  return `${y}-${String(m).padStart(2, '0')}-${String(d).padStart(2, '0')}`
}
