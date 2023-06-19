<script setup>
import { ref,getCurrentInstance } from 'vue'

defineProps({
  msg: String,
})

const count = ref(0)
const { proxy } = getCurrentInstance()
const $weui = proxy.$weui;

const test = () => {
  $weui.alert("服务器错误");
}
</script>

<template>
  <h1>{{ msg }}</h1>

  <div class="card">
    <button type="button" @click="count++">count is {{ count }}</button>
    <button @click="test">ddd</button>
  </div>
</template>

<style scoped>
.read-the-docs {
  color: #888;
}
</style>
