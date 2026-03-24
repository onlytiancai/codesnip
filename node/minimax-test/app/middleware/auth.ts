export default defineNuxtRouteMiddleware(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user) {
    return navigateTo('/login')
  }
})
