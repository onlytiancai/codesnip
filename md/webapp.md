初始化 web app

    pnpm create vite
    npm install tailwindcss @tailwindcss/vite


    % cat src/style.css 
    @import "tailwindcss";%                                                                                                                                             

    % cat vite.config.ts 
    import { defineConfig } from 'vite'
    import tailwindcss from '@tailwindcss/vite'
    import vue from '@vitejs/plugin-vue'

    export default defineConfig({
      plugins: [vue(), tailwindcss()],
    })

    % cat src/App.vue 
    <script setup lang="ts">
    </script>

    <template>
      <h1 class="text-3xl font-bold underline">
        Hello world!
      </h1>
    </template>

    <style scoped>
    </style>

    pnpm dev
