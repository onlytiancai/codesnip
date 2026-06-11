export default defineEventHandler(async (event) => {
  const query = getQuery(event)

  // Pagination
  const page = parseInt(query.page as string) || 1
  const limit = parseInt(query.limit as string) || 10
  const skip = (page - 1) * limit

  // Build filter conditions
  const where: any = {}

  if (query.status && query.status !== 'all') {
    where.status = query.status
  }

  if (query.plan && query.plan !== 'all') {
    where.plan = query.plan
  }

  if (query.search) {
    where.OR = [
      { orderNo: { contains: query.search as string } },
      { transactionId: { contains: query.search as string } },
      { User: { email: { contains: query.search as string } } },
      { User: { name: { contains: query.search as string } } }
    ]
  }

  // Get orders with pagination
  const [orders, total] = await Promise.all([
    prisma.order.findMany({
      where,
      skip,
      take: limit,
      orderBy: { createdAt: 'desc' },
      include: {
        User: {
          select: {
            id: true,
            email: true,
            name: true,
            avatar: true
          }
        }
      }
    }),
    prisma.order.count({ where })
  ])

  return {
    orders,
    pagination: {
      page,
      limit,
      total,
      totalPages: Math.ceil(total / limit)
    }
  }
})