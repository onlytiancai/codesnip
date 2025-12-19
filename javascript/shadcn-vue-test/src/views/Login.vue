<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { useAuth } from '@/composables/useAuth'

const { login } = useAuth()
</script>

<template>
  <div class="max-w-sm mx-auto mt-20 space-y-4">
    <h1 class="text-2xl font-bold">登录</h1>

    <Button class="w-full" @click="login('Huhao')">
      一键登录
    </Button>
  </div>
</template>
