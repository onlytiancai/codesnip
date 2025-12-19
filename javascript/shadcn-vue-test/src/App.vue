<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
</script>

<template>
  <div class="min-h-screen bg-background text-foreground">
    <!-- 顶部导航 -->
    <nav class="flex gap-4 border-b p-4">
      <RouterLink
        to="/"
        class="text-sm font-medium hover:underline"
      >
        首页
      </RouterLink>

      <RouterLink
        to="/about"
        class="text-sm font-medium hover:underline"
      >
        关于
      </RouterLink>
    </nav>

    <!-- 路由出口 -->
    <main class="p-6">
      <RouterView />
    </main>
  </div>
</template>
