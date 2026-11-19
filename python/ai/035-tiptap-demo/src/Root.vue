<script setup lang="ts">
import { RouterView } from 'vue-router'
import ThemeToggle from './components/ThemeToggle.vue'
</script>

<template>
  <RouterView />
  <!-- 主题切换按钮：整个站点只有一个，固定在右上角（登录页 / 登录后都在） -->
  <ThemeToggle />
</template>
