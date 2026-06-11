import { decryptResource, getWeChatPayConfig } from '../../utils/wechat-pay'

/**
 * WeChat Pay callback handler
 * This endpoint receives payment notifications from WeChat Pay
 */
export default defineEventHandler(async (event) => {
  const config = getWeChatPayConfig()

  // Read raw body for signature verification
  const rawBody = await readRawBody(event)

  if (!rawBody) {
    return {
      code: 'FAIL',
      message: 'Empty request body'
    }
  }

  let notification: any

  try {
    // Parse the notification
    notification = JSON.parse(rawBody)
  } catch (e) {
    return {
      code: 'FAIL',
      message: 'Invalid JSON'
    }
  }

  // Verify event type
  if (notification.event_type !== 'TRANSACTION.SUCCESS') {
    return {
      code: 'FAIL',
      message: 'Invalid event type'
    }
  }

  // Decrypt resource
  const resource = notification.resource
  let decryptedData: any

  try {
    const decryptedStr = decryptResource(
      resource.ciphertext,
      resource.associated_data || '',
      resource.nonce,
      config.apiV3Key
    )
    decryptedData = JSON.parse(decryptedStr)
  } catch (e) {
    console.error('Failed to decrypt WeChat Pay notification:', e)
    return {
      code: 'FAIL',
      message: 'Decryption failed'
    }
  }

  const {
    out_trade_no: orderNo,
    transaction_id: transactionId,
    trade_state: tradeState
  } = decryptedData

  // Find the order
  const order = await prisma.order.findUnique({
    where: { orderNo }
  })

  if (!order) {
    return {
      code: 'FAIL',
      message: 'Order not found'
    }
  }

  // Check if order is already paid
  if (order.status === 'paid') {
    return {
      code: 'SUCCESS',
      message: 'Order already processed'
    }
  }

  // Handle different trade states
  if (tradeState === 'SUCCESS') {
    // Payment successful
    const paidAt = new Date()

    // Update order status
    await prisma.order.update({
      where: { orderNo },
      data: {
        status: 'paid',
        transactionId,
        paidAt
      }
    })

    // Update user membership
    const startDate = new Date()
    let endDate: Date

    if (order.plan === 'pro') {
      // Monthly subscription
      endDate = new Date(startDate)
      endDate.setMonth(endDate.getMonth() + 1)
    } else {
      // Annual subscription
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

    console.log(`Payment successful for order ${orderNo}, user ${order.userId} upgraded to ${order.plan}`)

    return {
      code: 'SUCCESS',
      message: 'Success'
    }
  } else if (tradeState === 'CLOSED' || tradeState === 'PAYERROR') {
    // Payment failed
    await prisma.order.update({
      where: { orderNo },
      data: { status: 'failed' }
    })

    return {
      code: 'SUCCESS',
      message: 'Payment failed recorded'
    }
  }

  // Other states (NOTPAY, USERPAYING, etc.)
  return {
    code: 'SUCCESS',
    message: 'Notification received'
  }
})