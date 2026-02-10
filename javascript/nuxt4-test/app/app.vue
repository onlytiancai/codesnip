<script setup>
const { loggedIn, user, session, fetch, clear, openInPopup } = useUserSession()
const { data } = await useFetch('/api/hello')
async function submit () {
  const { body } = await $fetch('/api/submit', {
    method: 'post',
    body: { test: 123 },
  })
  data.value = body
}
</script>

<template>
   <button @click="submit">Submit</button>
   <pre>{{ data }}</pre>
  <div v-if="loggedIn">
    <h1>Welcome {{ user.login }}!</h1>
    <p>Logged in since {{ session.loggedInAt }}</p>
    <button @click="clear">Logout</button>
  </div>
  <div v-else>
    <h1>Not logged in</h1>
    <a href="/auth/github">Login with GitHub</a>
    <!-- or open the OAuth route in a popup -->
    <button @click="openInPopup('/auth/github')">Login with GitHub</button>
  </div>
</template>
