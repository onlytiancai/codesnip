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

onSlideLeave(() => stop())
onBeforeUnmount(() => stop())
</script>

<template>
  <!-- 无可视 UI，仅承载音频播放 -->
  <span style="display: none" />
</template>
