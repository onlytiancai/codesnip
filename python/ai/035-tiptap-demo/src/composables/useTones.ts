/**
 * 语气列表 —— 从后端拉，各 provider 可以有自己的中文 label。
 *
 * 拉不到就退回内置的那份（src/constants/tones.ts），不影响使用。
 */

import { ref } from 'vue'
import { TONES, type ToneId, type ToneOption } from '../constants/tones'
import { fetchConfig, fetchTones } from '../services/settings'

// 模块级单例 —— 翻译下拉和「改写语气」面板共用一份
const tones = ref<ToneOption[]>(TONES.map((t) => ({ ...t })))

export function useTones() {
  return { tones }
}

/** 拉当前生效 provider 的语气表。失败就保持内置的。 */
export async function loadTones(): Promise<void> {
  try {
    const cfg = await fetchConfig()
    const res = await fetchTones(cfg.activeProviderId)
    if (res.tones.length) {
      tones.value = res.tones.map((t) => ({
        id: t.id as ToneId,
        label: t.label,
      }))
    }
  } catch {
    // 后端没起来 / 老版本后端 → 用内置的
  }
}
