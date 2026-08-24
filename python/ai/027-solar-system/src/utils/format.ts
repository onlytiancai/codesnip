/**
 * 天文数值的儿童友好格式化（中文）。
 */

/** 距离（km）→ "1.5 亿 km" / "38.4 万 km" */
export function formatDistanceKm(km: number): string {
  if (km >= 1e8) return `${(km / 1e8).toFixed(km >= 1e9 ? 1 : 2)} 亿 km`
  if (km >= 1e4) return `${(km / 1e4).toFixed(1)} 万 km`
  return `${Math.round(km).toLocaleString('zh-CN')} km`
}

/** 周期（天）→ "88 天" / "11.9 年" */
export function formatPeriodDays(days: number): string {
  if (days <= 0) return '—'
  if (days >= 730) return `${(days / 365.25).toFixed(days >= 36525 ? 0 : 1)} 年`
  return `${days < 100 ? days.toFixed(days < 10 ? 1 : 0) : Math.round(days).toLocaleString('zh-CN')} 天`
}

/** 半径 km → "12,742 km" */
export function formatRadiusKm(km: number): string {
  return `${Math.round(km).toLocaleString('zh-CN')} km`
}

/** 速度倍数显示 */
export function formatTimeScale(v: number): string {
  if (v >= 1000) return '1000x'
  if (v >= 1) return `${v}x`
  return `${v}x`
}
