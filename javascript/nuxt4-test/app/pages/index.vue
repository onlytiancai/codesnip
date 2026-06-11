<script setup lang="ts">
const { loggedIn, user, session, fetch, clear, openInPopup } = useUserSession()
const { data } = await useFetch('/api/hello')
async function submit() {
  const { body } = await $fetch('/api/submit', {
    method: 'post',
    body: { test: 123 },
  })
  data.value = body
}

const username = ref('')
const password = ref('')
const errorMsg = ref('')

const login = async () => {
  const res: any = await $fetch('/api/login', {
    method: 'POST',
    body: {
      username: username.value,
      password: password.value
    }
  })

  if (res.code === 0) {
    fetch()
  } else {
    errorMsg.value = res.message
  }
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
    <div>
      <h2>用户登录</h2>
      <input v-model="username" placeholder="用户名" />
      <br />
      <input v-model="password" type="password" placeholder="密码" />
      <br />
      <button @click="login">登录</button>
      <p style="color:red">{{ errorMsg }}</p>
    </div>
  </div>
</template>
