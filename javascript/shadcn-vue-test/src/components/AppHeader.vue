<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { useUserStore } from '@/stores/user'
import ThemeToggle from '@/components/ThemeToggle.vue'
import { RouterLink } from 'vue-router'

const user = useUserStore()
</script>

<template>
  <header class="flex items-center justify-between border-b p-4">
    <nav class="flex gap-2 p-4 border-b">

      <Button as-child variant="ghost" >
        <RouterLink to="/" exact-active-class="bg-muted underline">首页</RouterLink>
      </Button>


      <Button as-child variant="ghost" >
        <RouterLink to="/about" exact-active-class="bg-muted underline">关于</RouterLink>
      </Button>
    </nav>

    <div class="flex items-center gap-2">
      <span class="text-sm text-muted-foreground">
        {{ user.loggedIn ? user.name : '未登录' }}
      </span>

      <Button
        v-if="user.loggedIn"
        size="sm"
        variant="outline"
        @click="user.logout"
      >
        退出
      </Button>

      <ThemeToggle />
    </div>
  </header>
</template>
