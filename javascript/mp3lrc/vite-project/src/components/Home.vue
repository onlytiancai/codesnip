<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const to = function (name, title) {
  router.push({
    name: 'list',
    query: {
      name: name,
      title: title,
    }
  })
}
const list = [
  {
    title: '新概念英语第一册',
    name: 'NCE1-英音-(MP3+LRC)',
  },
  {
    title: '新概念英语第二册',
    name: 'NCE2-英音-(MP3+LRC)',
  },
];

</script>
<template>
  <div class="weui-cells__title">新概念英语</div>
  <div class="weui-cells">
    <a @click="to(item.name, item.title)"  
       class="weui-cell  weui-cell_access" 
       href="javascript:void(0)" 
       v-for="item in list">
      <span class="weui-cell__bd">
        <span>{{ item.title }}</span>
      </span>
      <span class="weui-cell__ft">
      </span>
    </a>
  </div>
</template>