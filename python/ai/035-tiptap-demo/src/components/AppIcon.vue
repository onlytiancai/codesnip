<script setup lang="ts">
/**
 * 应用里的小图标（描边风格）—— 转发给 lucide-vue-next。
 *
 * 组件本身很薄：真正的映射表在 `appIcons.ts`，那里的 `AppIconName` 类型也能被别人 import。
 */
import { APP_ICONS, type AppIconName } from './appIcons'

const props = withDefaults(
  defineProps<{
    name: AppIconName
    /** 图标边长（px） */
    size?: number
  }>(),
  { size: 16 },
)
</script>

<template>
  <component :is="APP_ICONS[props.name]" :size="props.size" :stroke-width="2" />
</template>
