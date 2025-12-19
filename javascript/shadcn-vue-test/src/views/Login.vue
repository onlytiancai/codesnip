<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { useUserStore } from '@/stores/user'
import { useRouter, useRoute } from 'vue-router'

const user = useUserStore()
const router = useRouter()
const route = useRoute()

function handleLogin() {
  user.login('Huhao')

  // 登录成功后，跳回原本想访问的页面
  const redirect = route.query.redirect as string || '/'
  router.replace(redirect)
}
</script>

<template>
  <div class="max-w-sm mx-auto mt-20 space-y-4">
    <h1 class="text-2xl font-bold">登录</h1>

    <Button class="w-full" @click="handleLogin">
      一键登录
    </Button>
  </div>
</template>
