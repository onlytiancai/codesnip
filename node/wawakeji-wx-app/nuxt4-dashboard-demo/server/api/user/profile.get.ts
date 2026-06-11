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

  // Get user with stats
  const [user, readingHistoryCount, vocabularyCount, bookmarkCount, membership, preferences] = await Promise.all([
    prisma.user.findUnique({
      where: { id: userId },
      select: {
        id: true,
        email: true,
        name: true,
        avatar: true,
        bio: true,
        role: true,
        createdAt: true
      }
    }),
    prisma.readingHistory.count({ where: { userId } }),
    prisma.vocabulary.count({ where: { userId } }),
    prisma.bookmark.count({ where: { userId } }),
    prisma.membership.findUnique({ where: { userId } }),
    prisma.userPreferences.findUnique({ where: { userId } })
  ])

  if (!user) {
    throw createError({
      statusCode: 404,
      message: 'User not found'
    })
  }

  // Calculate reading streak
  const recentHistory = await prisma.readingHistory.findMany({
    where: {
      userId,
      lastReadAt: {
        gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) // Last 30 days
      }
    },
    select: { lastReadAt: true },
    orderBy: { lastReadAt: 'desc' }
  })

  // Calculate streak
  let streak = 0
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  for (let i = 0; i < 30; i++) {
    const checkDate = new Date(today)
    checkDate.setDate(checkDate.getDate() - i)
    const hasRead = recentHistory.some(h => {
      const readDate = new Date(h.lastReadAt)
      readDate.setHours(0, 0, 0, 0)
      return readDate.getTime() === checkDate.getTime()
    })
    if (hasRead) {
      streak++
    } else if (i > 0) {
      break
    }
  }

  // Calculate total reading time (estimated: 5 minutes per completed article)
  const completedArticles = await prisma.readingHistory.count({
    where: { userId, progress: 100 }
  })
  const inProgressArticles = await prisma.readingHistory.count({
    where: { userId, progress: { lt: 100 } }
  })
  const totalReadingMinutes = completedArticles * 8 + inProgressArticles * 4

  return {
    user,
    stats: {
      articlesRead: readingHistoryCount,
      vocabularyLearned: vocabularyCount,
      bookmarks: bookmarkCount,
      streak,
      totalReadingMinutes
    },
    membership: membership || { plan: 'free' },
    preferences: preferences || null
  }
})