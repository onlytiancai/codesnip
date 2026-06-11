export default defineNuxtRouteMiddleware(async (to) => {
  // Only protect /admin routes
  if (!to.path.startsWith('/admin')) {
    return
  }

  const { user } = await useUserSession()

  // Not logged in - redirect to login
  if (!user.value) {
    return navigateTo('/login', {
      query: { redirect: to.fullPath }
    })
  }

  // Not admin - redirect to home
  if (user.value.role !== 'ADMIN') {
    return navigateTo('/')
  }
})