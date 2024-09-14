<template>
  <img src="~/assets/logo@2x.png" alt="My Logo" />
  <header>
    <nav>
      <ul>
        <li><NuxtLink to="/about">关于</NuxtLink></li>
        <li><NuxtLink to="/countdown">倒计时</NuxtLink></li>
        <li><NuxtLink to="/">首页</NuxtLink></li>
      </ul>
    </nav>
  </header>
  <NuxtPage />
  <div class="static" :class="{ active: isActive, 'text-danger': hasError }">111</div>
  <div :class="classObject">222</div>
</template>
<style>
@import url("~/assets/style.css");
.static {
  height: 100px;
  width: 100px;
}
.active {
  background-color:bisque;
}
.text-danger {
  background-color: yellow;
}
.rotate-enter-active,
.rotate-leave-active {
  transition: all 0.4s;
}
.rotate-enter-from,
.rotate-leave-to {
  opacity: 0;
  transform: rotate3d(1, 1, 1, 15deg);
}
</style>
<script setup lang="ts">
const isActive = ref(true)
const hasError = ref(false)
const classObject = reactive({
  active: true,
  'text-danger': false
})
</script>
