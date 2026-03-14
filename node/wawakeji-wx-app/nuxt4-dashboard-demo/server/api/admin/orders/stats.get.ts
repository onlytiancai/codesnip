export default defineEventHandler(async () => {
  // Get order statistics
  const [
    totalOrders,
    pendingOrders,
    paidOrders,
    failedOrders,
    refundedOrders,
    totalRevenue,
    todayRevenue,
    monthRevenue,
    recentOrders
  ] = await Promise.all([
    // Total orders
    prisma.order.count(),
    // Pending orders
    prisma.order.count({ where: { status: 'pending' } }),
    // Paid orders
    prisma.order.count({ where: { status: 'paid' } }),
    // Failed orders
    prisma.order.count({ where: { status: 'failed' } }),
    // Refunded orders
    prisma.order.count({ where: { status: 'refunded' } }),
    // Total revenue (in cents)
    prisma.order.aggregate({
      where: { status: 'paid' },
      _sum: { amount: true }
    }),
    // Today's revenue
    prisma.order.aggregate({
      where: {
        status: 'paid',
        paidAt: {
          gte: new Date(new Date().setHours(0, 0, 0, 0))
        }
      },
      _sum: { amount: true }
    }),
    // This month's revenue
    prisma.order.aggregate({
      where: {
        status: 'paid',
        paidAt: {
          gte: new Date(new Date().getFullYear(), new Date().getMonth(), 1)
        }
      },
      _sum: { amount: true }
    }),
    // Recent 5 orders
    prisma.order.findMany({
      take: 5,
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
    })
  ])

  // Plan distribution
  const planDistribution = await prisma.order.groupBy({
    by: ['plan'],
    where: { status: 'paid' },
    _count: true,
    _sum: { amount: true }
  })

  return {
    totalOrders,
    pendingOrders,
    paidOrders,
    failedOrders,
    refundedOrders,
    totalRevenue: totalRevenue._sum.amount || 0,
    todayRevenue: todayRevenue._sum.amount || 0,
    monthRevenue: monthRevenue._sum.amount || 0,
    recentOrders,
    planDistribution
  }
})