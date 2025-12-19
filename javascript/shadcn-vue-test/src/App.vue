<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import { Button } from '@/components/ui/button'
</script>

<template>
  <div class="min-h-screen bg-background text-foreground">
    <!-- 顶部导航 -->
    <nav class="flex gap-2 p-4 border-b">

    <RouterLink to="/" custom v-slot="{ isExactActive, navigate, href }">
      <a :href="href" @click="navigate">
        <Button
          variant="ghost"
          :class="isExactActive ? 'bg-muted underline' : ''"
        >
          首页
        </Button>
      </a>
    </RouterLink>


      <Button as-child variant="ghost" >
        <RouterLink to="/about" exact-active-class="bg-muted underline">关于</RouterLink>
      </Button>
    </nav>

    <!-- 路由出口 -->
    <main class="p-6">
      <RouterView />
    </main>
  </div>
</template>
