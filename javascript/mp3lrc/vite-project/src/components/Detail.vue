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
    {{ title }}
</template>