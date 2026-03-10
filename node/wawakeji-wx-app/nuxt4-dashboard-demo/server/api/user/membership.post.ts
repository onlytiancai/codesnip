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

  if (!plan || !['pro', 'annual'].includes(plan)) {
    throw createError({
      statusCode: 400,
      message: 'Invalid plan'
    })
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
      plan: 'premium',
      startDate,
      endDate
    },
    create: {
      userId: session.user.id,
      plan: 'premium',
      startDate,
      endDate
    }
  })

  return { success: true, membership }
})