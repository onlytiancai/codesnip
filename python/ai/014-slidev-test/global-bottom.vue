<!--
  global-bottom.vue — Slidev 全局层，自动挂在每张 slide 底部（单实例）。
  在这里挂载 ClickAudio，注入 v-click → 音频 的映射。

  为什么硬编码 items：@slidev/parser 的 parseSync 依赖 Node fs，浏览器里跑不了；
  当前仅 2 段 v-click，硬编码最简单可靠。若 slides.md 增删 v-click，
  重跑 `pnpm tts:slides` 生成 mp3 后，手动同步此表即可。

  ClickAudio 由 Slidev 自动 import（components/ 目录）。
-->
<script setup lang="ts">
const items = [
  { slide: 1, click: 1, audio: '/audio/slide-1-click-1.mp3' },
  { slide: 1, click: 2, audio: '/audio/slide-1-click-2.mp3' },
  { slide: 1, click: 3, audio: '/audio/slide-1-click-3.mp3' },
  { slide: 1, click: 4, audio: '/audio/slide-1-click-4.mp3' },
  { slide: 1, click: 5, audio: '/audio/slide-1-click-5.mp3' },
]
</script>

<template>
  <ClickAudio :items="items" />
</template>
