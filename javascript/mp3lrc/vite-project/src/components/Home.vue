<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const to = function (name,title) {
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
        name:'NCE1-英音-(MP3+LRC)',
    },
    {
        title: '新概念英语第二册',
        name:'NCE2-英音-(MP3+LRC)',
    },
];

</script>
<template>
    <div class="weui-panel">
      <div class="weui-panel__hd">新概念英语</div>
      <div class="weui-panel__bd">
        <div class="weui-media-box weui-media-box_small-appmsg">
          <div class="weui-cells">
            <a @click="to(item.name, item.title)"
             class="weui-cell weui-cell_active weui-cell_access weui-cell_example"
             href="javascript:void(0)" 
              v-for="item in list">
              <div class="weui-cell__bd weui-cell_primary">
                <p>{{item.title}}</p>
              </div>
              <div class="weui-cell__ft"></div>
            </a>
          </div>
        </div>
      </div>
    </div>
  </template>