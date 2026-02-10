<script setup>
const { loggedIn, user, session, fetch, clear, openInPopup } = useUserSession()

async function refreshTokens() {
  await $fetch('/api/auth/weixin_refresh_tokens', { method: 'POST' })
  await fetch() // Reload session with new tokens
}
</script>

<template>
  <div v-if="loggedIn">
    <h1>Welcome !</h1>
    <p>Logged in since {{ session.loggedInAt }}</p>
    <pre>user:{{ user }}</pre>
    <pre>session:{{ session }}</pre>
    <button @click="refreshTokens">Refresh tokens</button>
    <button @click="fetch">Refresh session</button>
    <button @click="clear">Logout</button>
  </div>
  <div v-else>
    <h1>Not logged in</h1>
    <a href="/auth/weixin">Login with Weixin</a>
    <!-- or open the OAuth route in a popup -->
    <button @click="openInPopup('/auth/weixin')">Login with Weixin</button>
  </div>
</template>