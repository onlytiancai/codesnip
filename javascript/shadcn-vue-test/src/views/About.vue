<script setup lang="ts">
import { Button } from '@/components/ui/button'
import { useUserStore } from '@/stores/user'
import { useRouter } from 'vue-router'

const user = useUserStore()
const router = useRouter()

function handleLogout() {
  user.logout()
  router.replace('/login')
}

</script>

<template>
  <div class="space-y-4">
    <h1 class="text-2xl font-bold">About</h1>

    <p>只有登录后才能看到这个页面</p>

    <Button variant="destructive" @click="handleLogout">
      退出登录
    </Button>
  </div>
</template>
