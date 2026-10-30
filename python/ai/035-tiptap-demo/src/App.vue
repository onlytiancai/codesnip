<script setup lang="ts">
import { RouterView } from 'vue-router'
import AppSidebar from './components/AppSidebar.vue'

/**
 * 应用外壳：左侧栏 + 右侧内容区。
 *
 * 侧栏在所有路由常驻 —— 所以设置页里也能直接切翻译方向。
 */
</script>

<template>
  <div class="shell">
    <AppSidebar />
    <main class="shell__content">
      <RouterView />
    </main>
  </div>
</template>
