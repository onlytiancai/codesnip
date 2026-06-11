<template>
  <div>
    <h1>Users</h1>
    <ul v-if="users?.length">
      <li v-for="user in users" :key="user.id">
        {{ user.name }} ({{ user.email }})
      </li>
    </ul>
    <p v-else>No users yet.</p>
    <form @submit.prevent="addUser">
      <input v-model="newUser.name" placeholder="Name" required>
      <input v-model="newUser.email" placeholder="Email" required>
      <button type="submit">Add User</button>
    </form>
  </div>
</template>

<script setup>
const { data: users, refresh } = await useFetch('/api/users')
const newUser = ref({ name: '', email: '' })

async function addUser() {
  await $fetch('/api/users', {
    method: 'POST',
    body: newUser.value,
  })
  newUser.value = { name: '', email: '' } // Clear the form
  await refresh() // Refresh the user list
}
</script>