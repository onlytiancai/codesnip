<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { useAuth } from '@/composables/useAuth'

const { logout } = useAuth()
</script>

<template>
  <div class="space-y-4">
    <h1 class="text-2xl font-bold">About</h1>

    <p>只有登录后才能看到这个页面</p>

    <Button variant="destructive" @click="logout">
      退出登录
    </Button>
  </div>
</template>
