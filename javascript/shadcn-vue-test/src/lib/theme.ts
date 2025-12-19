import { ref } from 'vue'

const isDark = ref(
  document.documentElement.classList.contains('dark')
)

export function toggleTheme() {
  const root = document.documentElement
  isDark.value = root.classList.toggle('dark')
}

export function useTheme() {
  return { isDark, toggleTheme }
}
