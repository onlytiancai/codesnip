<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
import { Button } from '@/components/ui/button'
import AppHeader from './components/AppHeader.vue'
</script>

<template>  
  <div class="min-h-screen bg-background text-foreground">
    <AppHeader></AppHeader>    
    <main class="p-6">
      <RouterView />
    </main>
  </div>
</template>
