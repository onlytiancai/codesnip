<script setup lang="ts">
import HelloWorld from '../components/HelloWorld.vue'
import { Button } from '@/components/ui/button'

</script>

<template>
  <div>

    <div class="p-4 m-4 border rounded-lg">
      <Button variant="ghost" >Hello</Button>
    </div>

    <div class="p-4 m-4 border rounded-lg bg-muted">
      <div class="text-green-500 text-2xl font-bold">
        Tailwind OK
      </div>
    </div>


    
    <a href="https://vite.dev" target="_blank">
      <img src="/vite.svg" class="logo" alt="Vite logo" />
    </a>
    <a href="https://vuejs.org/" target="_blank">
      <img src="@/assets/vue.svg" class="logo vue" alt="Vue logo" />
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
