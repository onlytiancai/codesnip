<script setup>
import { ref, getCurrentInstance, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'

const { proxy } = getCurrentInstance()
const $weui = proxy.$weui;
let list = ref([]);
const route = useRoute()
const router = useRouter()
let book = route.query.name;


onMounted(async () => {
  const url = 'http://media.ihuhao.com/%E6%96%B0%E6%A6%82%E5%BF%B5%E8%8B%B1%E8%AF%AD/'+book+'/list.json';
  try {
    const res = await fetch(url);
    list.value = await res.json();
  } catch (error) {
    $weui.alert("服务器错误");
  }
});

const to = function (item) {
  router.push({
    name: 'detail',
    query: {
      title: item.mp3.slice(0, -4),
      book: book,
    }
  })
}

const getNo = function (item) {
  //  "001&002\uff0dExcuse Me.mp3"
  let pos = item.mp3.indexOf('\uff0d');
  return item.mp3.slice(0,pos).split('&')[0]
}
const getTitle = function(item) {
  let pos = item.mp3.indexOf('\uff0d');
  return item.mp3.slice(pos+1).slice(0, -4)
}
</script>

<template>
  <div class="weui-panel">
    <div class="weui-panel__hd"><a class="weui-link weui-wa-hotarea" href="#">返回</a> {{$route.query.title}}</div>
    <div class="weui-panel__bd">
      <div class="weui-media-box weui-media-box_small-appmsg">
        <div class="weui-cells">
          <a class="weui-cell weui-cell_active weui-cell_access weui-cell_example"
            @click="to(item)"  
            href="javascript:void(0)" 
            v-for="item in list">
            <div class="weui-cell__bd weui-cell_primary">
              <p><span class="no">{{ getNo(item) }}</span>{{ getTitle(item) }}</p>
            </div>
            <div class="weui-cell__ft"></div>
          </a>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.no {
  margin-right: 10px;
  font-weight: bold;
}
</style>
