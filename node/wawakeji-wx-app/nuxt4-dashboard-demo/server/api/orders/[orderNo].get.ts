import { queryOrder, getWeChatPayConfig } from '../../utils/wechat-pay'

/**
 * Query order status
 * Used by frontend to check payment result
 */
export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)

  if (!session?.user?.id) {
    throw createError({
      statusCode: 401,
      message: 'Unauthorized'
    })
  }

  const orderNo = getRouterParam(event, 'orderNo')

  if (!orderNo) {
    throw createError({
      statusCode: 400,
      message: 'Order number is required'
    })
  }

  // Find order in database
  const order = await prisma.order.findUnique({
    where: { orderNo }
  })

  if (!order) {
    throw createError({
      statusCode: 404,
      message: 'Order not found'
    })
  }

  // Verify order belongs to user
  if (order.userId !== session.user.id) {
    throw createError({
      statusCode: 403,
      message: 'Access denied'
    })
  }

  // If order is already paid, return status directly
  if (order.status === 'paid') {
    return {
      orderNo: order.orderNo,
      plan: order.plan,
      amount: order.amount,
      status: order.status,
      transactionId: order.transactionId,
      paidAt: order.paidAt,
      createdAt: order.createdAt
    }
  }

  // If order is pending, query WeChat Pay for latest status
  if (order.status === 'pending') {
    // Check if order expired
    if (order.expiredAt < new Date()) {
      await prisma.order.update({
        where: { orderNo },
        data: { status: 'failed' }
      })
      return {
        orderNo: order.orderNo,
        plan: order.plan,
        amount: order.amount,
        status: 'failed',
        message: 'Order expired',
        createdAt: order.createdAt
      }
    }

    // Query WeChat Pay for latest status
    try {
      const config = getWeChatPayConfig()
      const wechatOrder = await queryOrder(orderNo, config) as any

      if (wechatOrder.trade_state === 'SUCCESS') {
        // Payment successful - update database
        const paidAt = new Date()

        await prisma.order.update({
          where: { orderNo },
          data: {
            status: 'paid',
            transactionId: wechatOrder.transaction_id,
            paidAt
          }
        })

        // Update user membership
        const startDate = new Date()
        let endDate: Date

        if (order.plan === 'pro') {
          endDate = new Date(startDate)
          endDate.setMonth(endDate.getMonth() + 1)
        } else {
          endDate = new Date(startDate)
          endDate.setFullYear(endDate.getFullYear() + 1)
        }

        await prisma.membership.upsert({
          where: { userId: order.userId },
          update: {
            plan: order.plan === 'annual' ? 'annual' : 'premium',
            startDate,
            endDate
          },
          create: {
            userId: order.userId,
            plan: order.plan === 'annual' ? 'annual' : 'premium',
            startDate,
            endDate
          }
        })

        return {
          orderNo: order.orderNo,
          plan: order.plan,
          amount: order.amount,
          status: 'paid',
          transactionId: wechatOrder.transaction_id,
          paidAt,
          createdAt: order.createdAt
        }
      } else if (wechatOrder.trade_state === 'CLOSED' || wechatOrder.trade_state === 'PAYERROR') {
        await prisma.order.update({
          where: { orderNo },
          data: { status: 'failed' }
        })

        return {
          orderNo: order.orderNo,
          plan: order.plan,
          amount: order.amount,
          status: 'failed',
          message: wechatOrder.trade_state_desc || 'Payment failed',
          createdAt: order.createdAt
        }
      }

      // Still pending
      return {
        orderNo: order.orderNo,
        plan: order.plan,
        amount: order.amount,
        status: 'pending',
        tradeState: wechatOrder.trade_state,
        tradeStateDesc: wechatOrder.trade_state_desc,
        createdAt: order.createdAt,
        expiredAt: order.expiredAt
      }
    } catch (error) {
      console.error('Failed to query WeChat Pay order:', error)
      // Return current status from database
      return {
        orderNo: order.orderNo,
        plan: order.plan,
        amount: order.amount,
        status: order.status,
        createdAt: order.createdAt,
        expiredAt: order.expiredAt
      }
    }
  }

  // Return order status
  return {
    orderNo: order.orderNo,
    plan: order.plan,
    amount: order.amount,
    status: order.status,
    createdAt: order.createdAt
  }
})