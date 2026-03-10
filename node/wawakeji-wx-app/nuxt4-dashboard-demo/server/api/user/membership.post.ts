export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  const body = await readBody(event)
  const { plan } = body

  if (!plan || !['free', 'pro', 'annual'].includes(plan)) {
    throw createError({
      statusCode: 400,
      message: 'Invalid plan'
    })
  }

  // Handle downgrade to free
  if (plan === 'free') {
    await prisma.membership.deleteMany({
      where: { userId: session.user.id }
    })
    return { success: true, membership: { plan: 'free' } }
  }

  // Calculate end date
  const startDate = new Date()
  let endDate: Date

  if (plan === 'pro') {
    // Monthly subscription
    endDate = new Date(startDate)
    endDate.setMonth(endDate.getMonth() + 1)
  } else {
    // Annual subscription
    endDate = new Date(startDate)
    endDate.setFullYear(endDate.getFullYear() + 1)
  }

  // Update or create membership
  const membership = await prisma.membership.upsert({
    where: { userId: session.user.id },
    update: {
      plan: plan === 'annual' ? 'annual' : 'premium',
      startDate,
      endDate
    },
    create: {
      userId: session.user.id,
      plan: plan === 'annual' ? 'annual' : 'premium',
      startDate,
      endDate
    }
  })

  return { success: true, membership }
})