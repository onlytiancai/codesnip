<template>
  <div class="flex items-center gap-3">
    <!-- Main Play Button -->
    <UButton
      :icon="isPlaying ? 'i-lucide-pause' : 'i-lucide-play'"
      :color="isPlaying ? 'primary' : 'neutral'"
      size="sm"
      @click="togglePlay"
    />

    <!-- Progress Bar -->
    <div class="flex-1 h-1 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
      <div
        class="h-full bg-primary rounded-full transition-all duration-100"
        :style="{ width: progressPercent + '%' }"
      />
    </div>

    <!-- Time Display -->
    <span class="text-xs text-gray-500 min-w-[60px]">
      {{ currentTime }} / {{ totalTime }}
    </span>

    <!-- Speed Control -->
    <USelect
      v-model="playbackSpeed"
      :items="speedOptions"
      size="xs"
      class="w-20"
    />

    <!-- Play Mode Toggle -->
    <UButton
      v-if="showPlayMode"
      :icon="playMode === 'all' ? 'i-lucide-repeat' : 'i-lucide-repeat-1'"
      variant="ghost"
      size="xs"
      :color="playMode === 'all' ? 'primary' : 'neutral'"
      @click="togglePlayMode"
    />
  </div>
</template>

<script setup lang="ts">
interface AudioItem {
  en: string
  audio?: string
}

const props = withDefaults(defineProps<{
  items: AudioItem[]
  currentIndex?: number
  showPlayMode?: boolean
  autoPlay?: boolean
}>(), {
  currentIndex: 0,
  showPlayMode: true,
  autoPlay: false
})

const emit = defineEmits<{
  'update:currentIndex': [index: number]
  'play': [item: AudioItem]
  'pause': []
  'ended': []
}>()

const isPlaying = ref(false)
const currentIndex = ref(props.currentIndex)
const playbackSpeed = ref(1.0)
const playMode = ref<'single' | 'all'>('all')
const progressPercent = ref(0)
const currentTime = ref('0:00')
const totalTime = ref('0:00')

const speedOptions = [
  { label: '0.5x', value: 0.5 },
  { label: '0.75x', value: 0.75 },
  { label: '1.0x', value: 1.0 },
  { label: '1.25x', value: 1.25 },
  { label: '1.5x', value: 1.5 },
  { label: '2.0x', value: 2.0 }
]

// Web Speech API
const speechSynthesis = ref<SpeechSynthesis | null>(null)
const currentUtterance = ref<SpeechSynthesisUtterance | null>(null)
const estimatedDuration = ref(0)
const startTime = ref(0)
const progressInterval = ref<ReturnType<typeof setInterval> | null>(null)

const currentItem = computed(() => props.items[currentIndex.value])

const formatTime = (seconds: number): string => {
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

const estimateDuration = (text: string): number => {
  // Rough estimate: ~150 words per minute at 1x speed
  const words = text.split(' ').length
  return (words / 150) * 60
}

const playItem = (index: number) => {
  if (index >= props.items.length) {
    stopPlaying()
    emit('ended')
    return
  }

  currentIndex.value = index
  const item = props.items[index]

  if (!item.en) {
    // Skip empty items
    if (playMode.value === 'all') {
      playItem(index + 1)
    }
    return
  }

  // Use Web Speech API
  if ('speechSynthesis' in window) {
    speechSynthesis.value = window.speechSynthesis
    currentUtterance.value = new SpeechSynthesisUtterance(item.en)
    currentUtterance.value.lang = 'en-US'
    currentUtterance.value.rate = playbackSpeed.value

    // Estimate duration
    estimatedDuration.value = estimateDuration(item.en) / playbackSpeed.value
    totalTime.value = formatTime(estimatedDuration.value)
    startTime.value = Date.now()

    currentUtterance.value.onstart = () => {
      isPlaying.value = true
      emit('play', item)

      // Start progress updates
      progressInterval.value = setInterval(() => {
        const elapsed = (Date.now() - startTime.value) / 1000
        const progress = Math.min(100, (elapsed / estimatedDuration.value) * 100)
        progressPercent.value = progress
        currentTime.value = formatTime(elapsed)
      }, 100)
    }

    currentUtterance.value.onend = () => {
      if (progressInterval.value) {
        clearInterval(progressInterval.value)
      }
      progressPercent.value = 100
      currentTime.value = totalTime.value

      if (playMode.value === 'all' && currentIndex.value < props.items.length - 1) {
        // Play next item
        setTimeout(() => playItem(currentIndex.value + 1), 500)
      } else {
        isPlaying.value = false
        emit('ended')
      }
    }

    currentUtterance.value.onerror = () => {
      stopPlaying()
    }

    speechSynthesis.value.speak(currentUtterance.value)
  }
}

const stopPlaying = () => {
  if (progressInterval.value) {
    clearInterval(progressInterval.value)
  }

  if (speechSynthesis.value) {
    speechSynthesis.value.cancel()
  }

  isPlaying.value = false
  progressPercent.value = 0
  currentTime.value = '0:00'
}

const togglePlay = () => {
  if (isPlaying.value) {
    stopPlaying()
    emit('pause')
  } else {
    playItem(currentIndex.value)
  }
}

const togglePlayMode = () => {
  playMode.value = playMode.value === 'single' ? 'all' : 'single'
}

const play = (index?: number) => {
  if (index !== undefined) {
    currentIndex.value = index
  }
  playItem(currentIndex.value)
}

const pause = () => {
  stopPlaying()
}

const stop = () => {
  stopPlaying()
  currentIndex.value = 0
}

// Watch for speed changes
watch(playbackSpeed, (newSpeed) => {
  if (currentUtterance.value) {
    currentUtterance.value.rate = newSpeed
  }
})

// Watch for currentIndex prop changes
watch(() => props.currentIndex, (newIndex) => {
  currentIndex.value = newIndex
})

// Expose methods
defineExpose({
  play,
  pause,
  stop,
  isPlaying,
  currentIndex,
  playMode
})

// Cleanup
onUnmounted(() => {
  stopPlaying()
})
</script>