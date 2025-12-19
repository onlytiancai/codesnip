<script setup lang="ts">
import { Moon, Sun } from 'lucide-vue-next'
import { Button } from '@/components/ui/button'
import { useTheme } from '@/lib/theme'
import { computed } from 'vue'

const { isDark, toggleTheme } = useTheme()

const Icon = computed(() => (isDark.value ? Sun : Moon))
</script>

<template>
  <Button
    variant="ghost"
    size="icon"
    aria-label="Toggle theme"
    @click="toggleTheme"
  >
    <component :is="Icon" class="h-5 w-5" />
  </Button>
</template>
