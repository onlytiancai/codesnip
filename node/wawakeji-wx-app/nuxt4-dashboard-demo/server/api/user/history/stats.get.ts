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

  // Get total reading time (actual, not estimated)
  const history = await prisma.readingHistory.findMany({
    where: { userId },
    select: { readingTime: true }
  })

  // Sum up actual reading time from database
  const totalMinutes = history.reduce((sum, h) => sum + (h.readingTime || 0), 0)

  // Calculate streak - get distinct dates user has read
  const recentHistory = await prisma.readingHistory.findMany({
    where: {
      userId,
      lastReadAt: {
        gte: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000) // Look back 60 days
      }
    },
    select: { lastReadAt: true },
    orderBy: { lastReadAt: 'desc' }
  })

  // Get unique dates
  const readDates = new Set<string>()
  for (const h of recentHistory) {
    const date = new Date(h.lastReadAt)
    date.setHours(0, 0, 0, 0)
    readDates.add(date.toISOString().split('T')[0])
  }

  let streak = 0
  const today = new Date()
  today.setHours(0, 0, 0, 0)

  // Check if user read today, if not start counting from yesterday
  let startDate = new Date(today)
  const todayStr = today.toISOString().split('T')[0]
  if (!readDates.has(todayStr)) {
    // User hasn't read today yet, check if they read yesterday
    startDate.setDate(startDate.getDate() - 1)
    const yesterdayStr = startDate.toISOString().split('T')[0]
    if (!readDates.has(yesterdayStr)) {
      // User hasn't read today or yesterday, streak is 0
      streak = 0
    }
  }

  // Count consecutive days
  if (readDates.has(todayStr) || readDates.has(startDate.toISOString().split('T')[0])) {
    for (let i = 0; i < 60; i++) {
      const checkDate = new Date(startDate)
      checkDate.setDate(checkDate.getDate() - i)
      const dateStr = checkDate.toISOString().split('T')[0]
      if (readDates.has(dateStr)) {
        streak++
      } else {
        break
      }
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