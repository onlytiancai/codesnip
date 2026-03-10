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

  // Get total articles read (completed)
  const articlesRead = await prisma.readingHistory.count({
    where: { userId, progress: 100 }
  })

  // Get total reading time (estimated)
  const history = await prisma.readingHistory.findMany({
    where: { userId },
    include: {
      Article: {
        select: { content: true }
      }
    }
  })

  let totalMinutes = 0
  for (const h of history) {
    const wordCount = h.Article.content?.split(' ').length || 0
    const estimatedMinutes = Math.ceil(wordCount / 200 * (h.progress / 100))
    totalMinutes += estimatedMinutes
  }

  // Calculate streak
  const recentHistory = await prisma.readingHistory.findMany({
    where: {
      userId,
      lastReadAt: {
        gte: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
      }
    },
    select: { lastReadAt: true },
    orderBy: { lastReadAt: 'desc' }
  })

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

  // Get articles in progress
  const inProgress = await prisma.readingHistory.count({
    where: { userId, progress: { gt: 0, lt: 100 } }
  })

  // Get completed this week
  const weekAgo = new Date()
  weekAgo.setDate(weekAgo.getDate() - 7)
  const completedThisWeek = await prisma.readingHistory.count({
    where: {
      userId,
      completedAt: { gte: weekAgo }
    }
  })

  return {
    articlesRead,
    totalMinutes,
    streak,
    inProgress,
    completedThisWeek
  }
})