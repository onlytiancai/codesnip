<script setup>
import { onMounted, ref, watch } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
let title = ref('');
onMounted(() => {
    console.log(route.query);
    title.value = route.query.title
})

watch(
    () => route.query.title,
    newTitle => {
        title.value = newTitle
    }
)
</script>
<template>
    <header>{{ title }}</header>

    <div class="container">
        <ul class="wordList">
        </ul>
    </div>

    <footer>
        <div class="buttons">
            <select id="sel_subtitle">
                <option value="en">英文</option>
                <option value="cn">中文</option>
                <option value="en_cn">中英</option>
            </select>
        </div>
        <div class="player-container">
            <audio class="player" controls></audio>

        </div>

    </footer>
</template>