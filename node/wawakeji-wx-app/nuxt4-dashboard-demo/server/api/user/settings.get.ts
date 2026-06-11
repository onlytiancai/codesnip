export default defineEventHandler(async (event) => {
  // Get current user from session
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  const userId = session.user.id

  // Get or create user preferences
  let preferences = await prisma.userPreferences.findUnique({
    where: { userId }
  })

  if (!preferences) {
    // Create default preferences
    preferences = await prisma.userPreferences.create({
      data: { userId }
    })
  }

  // Parse interests JSON
  let interests: string[] = []
  if (preferences.interests) {
    try {
      interests = JSON.parse(preferences.interests)
    } catch {
      interests = []
    }
  }

  return {
    ...preferences,
    interests
  }
})