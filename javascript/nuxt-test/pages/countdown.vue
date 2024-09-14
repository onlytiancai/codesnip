<!-- components/Countdown.vue -->
<template>
    <div class="countdown-container">
      <h1>距离2025年还有</h1>
      <div class="countdown">
        <div class="time-unit" v-for="unit in timeUnits" :key="unit.label">
          <transition name="fade" @before-enter="beforeEnter" @enter="enter">
            <span class="number">{{ unit.value }}</span>
          </transition>
          <span class="label">{{ unit.label }}</span>
        </div>
      </div>
    </div>
  </template>
  
  <script setup>
  import { ref, onMounted } from 'vue';
  import dayjs from 'dayjs';
  
  const { data } = await useFetch('data.json')
  console.log(111, data)

  const targetDate = dayjs(data.date);
  
  // 定义倒计时变量
  const days = ref(0);
  const hours = ref(0);
  const minutes = ref(0);
  const seconds = ref(0);
  
  const updateCountdown = () => {
    const now = dayjs();
    const diff = targetDate.diff(now);
  
    days.value = Math.floor(diff / (1000 * 60 * 60 * 24));
    hours.value = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
    minutes.value = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    seconds.value = Math.floor((diff % (1000 * 60)) / 1000);
  };
  
  const timeUnits = ref([
    { label: '天', value: days.value },
    { label: '小时', value: hours.value },
    { label: '分钟', value: minutes.value },
    { label: '秒', value: seconds.value }
  ]);
  
  onMounted(() => {
    updateCountdown();
    setInterval(() => {
      updateCountdown();
      timeUnits.value = [
        { label: '天', value: days.value },
        { label: '小时', value: hours.value },
        { label: '分钟', value: minutes.value },
        { label: '秒', value: seconds.value }
      ];
    }, 1000);
  });
  
  // 动画钩子函数
  const beforeEnter = (el) => {
    el.style.opacity = 0;
  };
  
  const enter = (el, done) => {
    el.offsetHeight; // trigger reflow
    el.style.transition = 'opacity 0.5s';
    el.style.opacity = 1;
    done();
  };
  </script>
  
  <style scoped>
  .countdown-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    background: #282c34;
    color: #fff;
    font-family: Arial, sans-serif;
  }
  
  h1 {
    font-size: 2em;
    margin-bottom: 20px;
  }
  
  .countdown {
    display: flex;
  }
  
  .time-unit {
    display: flex;
    flex-direction: column;
    align-items: center;
    margin: 0 10px;
    background: #444;
    padding: 20px;
    border-radius: 8px;
    position: relative;
    overflow: hidden;
  }
  
  .number {
    font-size: 3em;
    transition: transform 0.5s ease-in-out;
  }
  
  .label {
    font-size: 1.2em;
  }
  
  .fade-enter-active,
  .fade-leave-active {
    transition: opacity 0.5s;
  }
  
  .fade-enter, .fade-leave-to /* .fade-leave-active in <2.1.8 */ {
    opacity: 0;
  }
  </style>
  