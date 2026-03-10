export default defineNuxtRouteMiddleware(async (to) => {
  // Only protect /user routes
  if (!to.path.startsWith('/user')) {
    return
  }

  const { loggedIn } = await useUserSession()

  // Not logged in - redirect to login
  if (!loggedIn.value) {
    return navigateTo('/login', {
      query: { redirect: to.fullPath }
    })
  }
})