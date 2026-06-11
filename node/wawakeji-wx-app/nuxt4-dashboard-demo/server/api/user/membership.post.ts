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

  // Only allow downgrade to free through this endpoint
  // Upgrades should go through WeChat Pay
  if (plan !== 'free') {
    throw createError({
      statusCode: 400,
      message: 'Please use WeChat Pay to upgrade your membership'
    })
  }

  // Handle downgrade to free
  await prisma.membership.deleteMany({
    where: { userId: session.user.id }
  })

  return { success: true, membership: { plan: 'free' } }
})