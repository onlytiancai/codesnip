<script setup lang="ts">
import HelloWorld from './components/HelloWorld.vue'
import { Button } from '@/components/ui/button'
import { toggleDark } from '@/lib/theme'
</script>

<template>
  <div>
    <div class="text-6xl font-black underline">
      Tailwind v4 OK
    </div>
    <div class="text-primary">333</div>
    <div class="text-destructive">444</div>
    <div class="bg-muted text-muted-foreground">555</div>
    <Button @click="toggleDark">Hello shadcn-vue</Button>
    <a href="https://vite.dev" target="_blank">
      <img src="/vite.svg" class="logo" alt="Vite logo" />
    </a>
    <a href="https://vuejs.org/" target="_blank">
      <img src="./assets/vue.svg" class="logo vue" alt="Vue logo" />
    </a>
  </div>
  <HelloWorld msg="Vite + Vue" />
</template>

<style scoped>
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
