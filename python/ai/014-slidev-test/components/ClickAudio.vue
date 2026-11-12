<!--
  ClickAudio.vue — 订阅当前 slide 的 click 数，v-click 触发时自动播放对应旁白。

  数据源说明（关键）：
    在 global-bottom 这类全局层里，useSlideContext() 的 $clicks 是「死」的
    （不随按键更新，只有挂载时的初值）。必须用 useNav() 的 currentPage / clicks，
    它们才是随导航实时更新的全局响应式值。

  逻辑：
    - watch([currentPage, clicks])，找 items 里 slide===page && click===click 的项
    - currentKey 守门：同一 (page,click) 不重复播
    - 无匹配（标题页 click 0 或切走到别的 slide）→ stop() 静默
    - autoplay 被拦截时 catch + warn；首次用户按键即用户手势，浏览器随后放行
-->
<script setup lang="ts">
import { watch, onBeforeUnmount } from 'vue'
import { useNav, onSlideLeave } from '@slidev/client'

type AudioItem = { slide: number; click: number; audio: string }

const props = defineProps<{ items: AudioItem[] }>()

const { currentPage, clicks } = useNav()

let el: HTMLAudioElement | null = null
let currentKey = ''

function stop() {
  if (el) {
    el.pause()
    el.currentTime = 0
    el = null
  }
  currentKey = ''
}

function playFor(page: number, click: number) {
  const item = props.items.find((it) => it.slide === page && it.click === click)
  if (!item) {
    // 标题页 click 0，或切走到别的 slide（page 变 / click 归零）→ 停并静默
    stop()
    return
  }
  const key = `${page}:${click}`
  if (key === currentKey) return // 同一 click 不重播
  currentKey = key

  if (el) {
    el.pause()
    el.currentTime = 0
  }
  el = new Audio(item.audio)
  el.play().catch((err) => {
    console.warn('[ClickAudio] autoplay 被拦截，等待用户手势：', err?.name, err?.message ?? err)
  })
}

watch(
  [currentPage, clicks],
  ([page, click]) => playFor(page as number, click as number),
  { immediate: true },
)

// 兜底：global-bottom.vue 是异步 fetch manifest.json 后才把 items 传过来。
// 当 items 从 [] 变成有内容时，items 不会触发 [page, clicks] watch，
// 需要显式按当前 (page, click) 重判一次，让加载完成前已点出的 click 也能补播。
watch(
  () => props.items,
  () => playFor(currentPage.value as number, clicks.value as number),
  { deep: true },
)

// 独立 watch：把当前 click 序号和「当前页所有 [data-anim-ms] 的最大值」
// 暴露到 window，供 record-video.ts 读取以决定点下一次前的等待时长。
// 不耦合到 playFor（有些 v-click 无音频也得写 __lastClickAnimMs）。
watch(
  [currentPage, clicks],
  ([page, click]) => {
    if (typeof window === 'undefined') return
    ;(window as any).__clicks = click
    ;(window as any).__page = page

    let maxMs = 0
    document.querySelectorAll('[data-anim-ms]').forEach((el) => {
      const n = Number(el.getAttribute('data-anim-ms') || '0')
      if (Number.isFinite(n) && n > maxMs) maxMs = n
    })
    ;(window as any).__lastClickAnimMs = maxMs
  },
  { immediate: true },
)

onSlideLeave(() => stop())
onBeforeUnmount(() => stop())
</script>

<template>
  <!-- 无可视 UI，仅承载音频播放 -->
  <span style="display: none" />
</template>
