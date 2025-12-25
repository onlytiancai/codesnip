import { computed } from 'vue'
import { useUserStore } from '@/stores/user'
import { useRouter, useRoute } from 'vue-router'

export function useAuth() {
  const user = useUserStore()
  const router = useRouter()
  const route = useRoute()

  const isLoggedIn = computed(() => user.loggedIn)
  const username = computed(() => user.name)

  function login(name: string) {
    user.login(name)

    const redirect = route.query.redirect as string || '/'
    router.replace(redirect)
  }

  function logout() {
    user.logout()
    router.replace('/login')
  }

  return {
    isLoggedIn,
    username,
    login,
    logout,
  }
}
