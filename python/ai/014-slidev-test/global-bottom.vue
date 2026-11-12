<!--
  global-bottom.vue — Slidev 全局层，自动挂在每张 slide 底部（单实例）。
  在这里挂载 ClickAudio，把 (slide, click) → audio 的映射塞给它。

  映射从 `/audio/manifest.json` 读，由 `pnpm tts:slides` 生成。
  增删 slides.md 里的 v-click 旁白后，重跑 `pnpm tts:slides` 即可自动同步，
  无需再手动改这个文件。manifest 缺失时 items 为空 → ClickAudio 静默无音频。

  dev 浏览器已加载过的 manifest 会被内存缓存；如果改了 slides.md 并重跑了
  tts:slides，浏览器需要 hard reload 才能拉到新 manifest（fetch 加 no-cache
  保证同会话内不会被 304 缓存）。
-->
<script setup lang="ts">
import { ref, watch, onMounted } from 'vue'

type AudioItem = { slide: number; click: number; audio: string }

const items = ref<AudioItem[]>([])
const loadError = ref<string>('')

onMounted(async () => {
  try {
    const r = await fetch('/audio/manifest.json', { cache: 'no-cache' })
    if (!r.ok) {
      loadError.value = `HTTP ${r.status}`
      return
    }
    const data = await r.json()
    if (!Array.isArray(data?.items)) {
      loadError.value = 'items 不是数组'
      return
    }
    items.value = data.items as AudioItem[]
    console.log(`[ClickAudio 映射] 已加载 ${items.value.length} 条`)
  } catch (e) {
    loadError.value = (e as Error).message
    console.warn('[ClickAudio 映射] 加载失败：', loadError.value)
  }
})

// 调试用：在控制台暴露，方便排查
if (typeof window !== 'undefined') {
  watch(items, (v) => {
    ;(window as any).__audioItems = v
  }, { immediate: true })
}
</script>

<template>
  <ClickAudio :items="items" />
</template>