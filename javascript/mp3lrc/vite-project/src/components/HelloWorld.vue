<script setup>
import { ref, getCurrentInstance, onMounted } from 'vue'
import { useRouter } from 'vue-router'
const router = useRouter()

const to = function (title) {
  router.push({
    name: 'detail',
    query: {
      title: title
    }
  })
}

defineProps({
  msg: String,
})

const { proxy } = getCurrentInstance()
const $weui = proxy.$weui;
let list = ref([]);

onMounted(async () => {
  const url = 'list.json';
  try {
    const res = await fetch(url);
    list.value = await res.json();
  } catch (error) {
    $weui.alert("服务器错误");
  }
});


</script>

<template>
  <div class="weui-panel">
    <div class="weui-panel__hd">{{ msg }}</div>
    <div class="weui-panel__bd">
      <div class="weui-media-box weui-media-box_small-appmsg">
        <div class="weui-cells">
          <a @click="to(item.mp3)" class="weui-cell weui-cell_active weui-cell_access weui-cell_example"
            href="javascript:void(0)" v-for="item in list">
            <div class="weui-cell__bd weui-cell_primary">
              <p><span class="no">{{ item.mp3.slice(0, 3) }}</span>{{ item.mp3.slice(8).slice(0, -4) }}</p>
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
