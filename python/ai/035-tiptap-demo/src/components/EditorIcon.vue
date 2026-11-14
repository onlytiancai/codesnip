<script setup lang="ts">
/**
 * 工具栏上的小图标（描边风格，16px）。
 *
 * 为什么不用 emoji / 文字：
 *  - emoji 在不同系统上是彩色大图，尺寸对不齐
 *  - 文字按钮（「清空」「复制」）占横向空间，半栏宽的工具栏挤不下
 * 所以统一用内联 SVG —— 跟着 currentColor 变色，跟工具栏风格一致。
 */
defineProps<{
  name: 'undo' | 'redo' | 'copy' | 'trash' | 'check'
}>()
</script>

<template>
  <svg
    class="editor-icon"
    width="16"
    height="16"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
    focusable="false"
  >
    <!-- 撤销：往左拐的箭头 -->
    <template v-if="name === 'undo'">
      <path d="M9 14 4 9l5-5" />
      <path d="M4 9h10a5 5 0 0 1 0 10h-3" />
    </template>

    <!-- 重做：往右拐的箭头 -->
    <template v-else-if="name === 'redo'">
      <path d="m15 14 5-5-5-5" />
      <path d="M20 9H10a5 5 0 0 0 0 10h3" />
    </template>

    <!-- 复制：两张叠起来的纸 -->
    <template v-else-if="name === 'copy'">
      <rect x="9" y="9" width="11" height="11" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </template>

    <!-- 清空：垃圾桶 -->
    <template v-else-if="name === 'trash'">
      <path d="M3 6h18" />
      <path d="M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </template>

    <!-- 复制成功 -->
    <template v-else>
      <path d="M20 6 9 17l-5-5" />
    </template>
  </svg>
</template>
