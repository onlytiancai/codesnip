import { defineStore } from 'pinia'

export const useUserStore = defineStore('user', {
  state: () => ({
    name: 'Guest',
    loggedIn: false,
  }),
  actions: {
    login(name: string) {
      this.name = name
      this.loggedIn = true
    },
    logout() {
      this.name = 'Guest'
      this.loggedIn = false
    },
  },
})
