
export default defineEventHandler(async (event) => {
  const url = getRequestURL(event)

  // Protect all /api/admin routes
  if (url.pathname.startsWith('/api/admin')) {
    const session = await getUserSession(event)

    if (!session?.user) {
      throw createError({
        statusCode: 401,
        statusMessage: 'Unauthorized',
        message: 'You must be logged in to access this resource'
      })
    }

    if (session.user.role !== 'ADMIN') {
      throw createError({
        statusCode: 403,
        statusMessage: 'Forbidden',
        message: 'You do not have permission to access this resource'
      })
    }

    // Add user to event context for use in handlers
    event.context.user = session.user
  }
})