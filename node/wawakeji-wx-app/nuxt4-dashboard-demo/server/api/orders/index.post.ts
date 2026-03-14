import { z } from 'zod'
import {
  getWeChatPayConfig,
  createJSAPIPayment,
  generateOrderNo,
  PLAN_PRICES,
  type PlanType
} from '../../utils/wechat-pay'

const createOrderSchema = z.object({
  plan: z.enum(['pro', 'annual'])
})

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  // Validate request body
  const body = await readValidatedBody(event, createOrderSchema.parse)
  const { plan } = body

  // Get user's openid from Account table
  const account = await prisma.account.findFirst({
    where: {
      userId: session.user.id,
      provider: 'wechat_mp'
    }
  })

  if (!account?.providerAccountId) {
    throw createError({
      statusCode: 400,
      message: 'WeChat account not linked. Please login with WeChat first.'
    })
  }

  const openid = account.providerAccountId
  const amount = PLAN_PRICES[plan as PlanType]

  // Check if user already has a pending order
  const existingPendingOrder = await prisma.order.findFirst({
    where: {
      userId: session.user.id,
      status: 'pending',
      expiredAt: {
        gt: new Date()
      }
    }
  })

  if (existingPendingOrder) {
    // Return existing order instead of creating new one
    const config = getWeChatPayConfig()
    try {
      const paymentParams = await createJSAPIPayment(
        existingPendingOrder.orderNo,
        `English Reading App - ${plan === 'annual' ? 'Annual' : 'Pro Monthly'}`,
        amount,
        openid,
        config
      )
      return {
        success: true,
        orderNo: existingPendingOrder.orderNo,
        paymentParams
      }
    } catch (error) {
      // If payment creation fails, close old order and create new one
      await prisma.order.update({
        where: { id: existingPendingOrder.id },
        data: { status: 'failed' }
      })
    }
  }

  // Generate order number
  const orderNo = generateOrderNo()

  // Calculate expiration time (30 minutes from now)
  const expiredAt = new Date(Date.now() + 30 * 60 * 1000)

  // Create order in database
  const order = await prisma.order.create({
    data: {
      orderNo,
      userId: session.user.id,
      plan,
      amount,
      expiredAt
    }
  })

  // Get WeChat Pay config and create payment
  const config = getWeChatPayConfig()

  try {
    const paymentParams = await createJSAPIPayment(
      orderNo,
      `English Reading App - ${plan === 'annual' ? 'Annual' : 'Pro Monthly'}`,
      amount,
      openid,
      config
    )

    return {
      success: true,
      orderNo,
      paymentParams
    }
  } catch (error: any) {
    // Update order status to failed
    await prisma.order.update({
      where: { id: order.id },
      data: { status: 'failed' }
    })

    console.error('WeChat Pay order creation failed:', error)

    throw createError({
      statusCode: 500,
      message: error?.data?.message || 'Failed to create WeChat Pay order'
    })
  }
})