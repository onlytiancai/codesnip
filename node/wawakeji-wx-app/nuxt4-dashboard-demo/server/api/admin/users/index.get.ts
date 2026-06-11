
export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 10
  const skip = (page - 1) * limit

  // Build filter conditions
  const where: any = {}

  if (query.role && query.role !== 'all') {
    where.role = query.role
  }

  if (query.search) {
    where.OR = [
      { name: { contains: query.search as string } },
      { email: { contains: query.search as string } }
    ]
  }

  // Get users with pagination
  const [users, total] = await Promise.all([
    prisma.user.findMany({
      where,
      skip,
      take: limit,
      orderBy: { createdAt: 'desc' },
      select: {
        id: true,
        email: true,
        name: true,
        avatar: true,
        role: true,
        createdAt: true,
        updatedAt: true,
        _count: {
          select: { Article: true }
        }
      }
    }),
    prisma.user.count({ where })
  ])

  return {
    users: users.map(user => ({
      ...user,
      articleCount: user._count.Article
    })),
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})