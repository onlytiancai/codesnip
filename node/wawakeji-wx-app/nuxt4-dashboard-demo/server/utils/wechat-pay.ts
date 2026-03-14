import crypto from 'crypto'

/**
 * WeChat Pay configuration
 */
export interface WeChatPayConfig {
  mchId: string           // Merchant ID
  apiV3Key: string        // API v3 key
  serialNo: string        // Certificate serial number
  privateKey: string      // Merchant private key (PEM format)
  appId: string           // WeChat AppID (Official Account or Mini Program)
  notifyUrl: string       // Payment callback URL
}

/**
 * Get WeChat Pay configuration from runtime config
 */
export function getWeChatPayConfig(): WeChatPayConfig {
  const config = useRuntimeConfig()
  return {
    mchId: config.wechatPay?.mchId || '',
    apiV3Key: config.wechatPay?.apiV3Key || '',
    serialNo: config.wechatPay?.serialNo || '',
    privateKey: config.wechatPay?.privateKey || '',
    appId: config.oauth?.wechatMp?.appid || '',
    notifyUrl: config.wechatPay?.notifyUrl || '',
  }
}

/**
 * Generate random string
 */
export function generateNonceStr(length = 32): string {
  const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'
  let result = ''
  for (let i = 0; i < length; i++) {
    result += chars.charAt(Math.floor(Math.random() * chars.length))
  }
  return result
}

/**
 * Generate merchant order number
 * Format: timestamp + random 6 digits
 */
export function generateOrderNo(): string {
  const timestamp = Date.now().toString()
  const random = Math.floor(Math.random() * 1000000).toString().padStart(6, '0')
  return `${timestamp}${random}`
}

/**
 * Create signature for WeChat Pay API v3
 */
export function createSignature(
  method: string,
  url: string,
  timestamp: string,
  nonceStr: string,
  body: string,
  privateKey: string
): string {
  const message = `${method}\n${url}\n${timestamp}\n${nonceStr}\n${body}\n`
  const sign = crypto.createSign('RSA-SHA256')
  sign.update(message)
  return sign.sign(privateKey, 'base64')
}

/**
 * Build Authorization header for WeChat Pay API v3
 */
export function buildAuthorization(
  mchId: string,
  serialNo: string,
  nonceStr: string,
  timestamp: string,
  signature: string
): string {
  return `WECHATPAY2-SHA256-RSA2048 mchid="${mchId}",nonce_str="${nonceStr}",signature="${signature}",timestamp="${timestamp}",serial_no="${serialNo}"`
}

/**
 * Verify WeChat Pay callback signature
 */
export function verifySignature(
  timestamp: string,
  nonceStr: string,
  body: string,
  signature: string,
  publicKey: string
): boolean {
  const message = `${timestamp}\n${nonceStr}\n${body}\n`
  const verify = crypto.createVerify('RSA-SHA256')
  verify.update(message)
  return verify.verify(publicKey, signature, 'base64')
}

/**
 * Decrypt AES-256-GCM encrypted data from WeChat Pay callback
 */
export function decryptResource(
  ciphertext: string,
  associatedData: string,
  nonce: string,
  apiV3Key: string
): string {
  const key = Buffer.from(apiV3Key, 'utf8')
  const iv = Buffer.from(nonce, 'utf8')
  const authTag = Buffer.from(ciphertext.slice(-16), 'base64')
  const data = Buffer.from(ciphertext.slice(0, -16), 'base64')

  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv)
  decipher.setAuthTag(authTag)
  decipher.setAAD(Buffer.from(associatedData, 'utf8'))

  let decrypted = decipher.update(data, undefined, 'utf8')
  decrypted += decipher.final('utf8')

  return decrypted
}

/**
 * JSAPI payment request response
 */
export interface JSAPIPaymentResponse {
  appId: string
  timeStamp: string
  nonceStr: string
  package: string
  signType: string
  paySign: string
}

/**
 * Create JSAPI payment params for WeChat JS-SDK
 */
export async function createJSAPIPayment(
  orderNo: string,
  description: string,
  amount: number,
  openid: string,
  config: WeChatPayConfig
): Promise<JSAPIPaymentResponse> {
  const { mchId, appId, serialNo, privateKey, notifyUrl } = config

  const timestamp = Math.floor(Date.now() / 1000).toString()
  const nonceStr = generateNonceStr()

  const body = {
    appid: appId,
    mchid: mchId,
    description,
    out_trade_no: orderNo,
    notify_url: notifyUrl,
    amount: {
      total: amount,
      currency: 'CNY'
    },
    payer: {
      openid
    }
  }

  const bodyStr = JSON.stringify(body)
  const url = '/v3/pay/transactions/jsapi'
  const signature = createSignature('POST', url, timestamp, nonceStr, bodyStr, privateKey)
  const authorization = buildAuthorization(mchId, serialNo, nonceStr, timestamp, signature)

  // Call WeChat Pay API
  const response = await $fetch('https://api.mch.weixin.qq.com/v3/pay/transactions/jsapi', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': authorization
    },
    body: bodyStr
  }) as any

  // Generate JS-SDK payment params
  const prepayId = response.prepay_id
  const payTimestamp = Math.floor(Date.now() / 1000).toString()
  const payNonceStr = generateNonceStr()
  const packageStr = `prepay_id=${prepayId}`

  // Sign for JS-SDK
  const payMessage = `${appId}\n${payTimestamp}\n${payNonceStr}\n${packageStr}\n`
  const paySignObj = crypto.createSign('RSA-SHA256')
  paySignObj.update(payMessage)
  const paySign = paySignObj.sign(privateKey, 'base64')

  return {
    appId,
    timeStamp: payTimestamp,
    nonceStr: payNonceStr,
    package: packageStr,
    signType: 'RSA',
    paySign
  }
}

/**
 * Query order status from WeChat Pay
 */
export async function queryOrder(
  orderNo: string,
  config: WeChatPayConfig
): Promise<any> {
  const { mchId, appId, serialNo, privateKey } = config

  const timestamp = Math.floor(Date.now() / 1000).toString()
  const nonceStr = generateNonceStr()
  const url = `/v3/pay/transactions/out-trade-no/${orderNo}?mchid=${mchId}`

  const signature = createSignature('GET', url, timestamp, nonceStr, '', privateKey)
  const authorization = buildAuthorization(mchId, serialNo, nonceStr, timestamp, signature)

  const response = await $fetch(`https://api.mch.weixin.qq.com${url}`, {
    method: 'GET',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': authorization
    }
  })

  return response
}

/**
 * Close order
 */
export async function closeOrder(
  orderNo: string,
  config: WeChatPayConfig
): Promise<void> {
  const { mchId, appId, serialNo, privateKey } = config

  const timestamp = Math.floor(Date.now() / 1000).toString()
  const nonceStr = generateNonceStr()
  const url = `/v3/pay/transactions/out-trade-no/${orderNo}/close`

  const body = JSON.stringify({ mchid: mchId })
  const signature = createSignature('POST', url, timestamp, nonceStr, body, privateKey)
  const authorization = buildAuthorization(mchId, serialNo, nonceStr, timestamp, signature)

  await $fetch(`https://api.mch.weixin.qq.com${url}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'application/json',
      'Authorization': authorization
    },
    body
  })
}

/**
 * Plan pricing (amount in cents/分)
 */
export const PLAN_PRICES = {
  pro: 6800,      // ¥68
  annual: 46800   // ¥468
} as const

export type PlanType = keyof typeof PLAN_PRICES

/**
 * Get plan display name
 */
export function getPlanDisplayName(plan: string): string {
  switch (plan) {
    case 'pro':
      return 'Pro Monthly'
    case 'annual':
      return 'Pro Annual'
    default:
      return plan
  }
}