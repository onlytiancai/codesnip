import { defineStore } from 'pinia'

// 定义认证状态类型
interface AuthState {
  password: string
}

// 定义认证store
export const useAuthStore = defineStore('auth', {
  state: (): AuthState => ({
    password: ''
  }),
  actions: {
    setPassword(newPassword: string) {
      this.password = newPassword
    },
    clearPassword() {
      this.password = ''
    }
  }
})
