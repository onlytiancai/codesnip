/**
 * WeChat utility functions
 */

/**
 * Check if the request is from WeChat browser
 */
export function isWeChatBrowser(userAgent: string): boolean {
  return /MicroMessenger/i.test(userAgent)
}

/**
 * WeChat Open Platform OAuth URLs
 */
export const WECHAT_OPEN_URLS = {
  // QR code login for PC
  authorize: 'https://open.weixin.qq.com/connect/qrconnect',
  // Token endpoint
  token: 'https://api.weixin.qq.com/sns/oauth2/access_token',
  // User info endpoint
  userinfo: 'https://api.weixin.qq.com/sns/userinfo',
}

/**
 * WeChat Official Account (公众号) OAuth URLs
 */
export const WECHAT_MP_URLS = {
  // OAuth authorize for WeChat in-app browser
  authorize: 'https://open.weixin.qq.com/connect/oauth2/authorize',
  // Token endpoint
  token: 'https://api.weixin.qq.com/sns/oauth2/access_token',
  // User info endpoint
  userinfo: 'https://api.weixin.qq.com/sns/userinfo',
}

/**
 * WeChat user info response
 */
export interface WeChatUserInfo {
  openid?: string
  unionid?: string
  nickname?: string
  sex?: number
  province?: string
  city?: string
  country?: string
  headimgurl?: string
  privilege?: string[]
  // Error fields (returned when there's an error)
  errcode?: number
  errmsg?: string
}

/**
 * WeChat access token response
 */
export interface WeChatTokenResponse {
  access_token: string
  expires_in: number
  refresh_token: string
  openid: string
  unionid?: string
  scope: string
  errcode?: number
  errmsg?: string
}

/**
 * Exchange code for access token (Open Platform)
 */
export async function getWeChatOpenToken(
  code: string,
  appId: string,
  appSecret: string
): Promise<WeChatTokenResponse> {
  const url = `${WECHAT_OPEN_URLS.token}?appid=${appId}&secret=${appSecret}&code=${code}&grant_type=authorization_code`

  const response = await $fetch(url)

  // WeChat API returns JSON, but $fetch might return it as string
  // Need to parse it manually if it's a string
  if (typeof response === 'string') {
    return JSON.parse(response)
  }

  return response as WeChatTokenResponse
}

/**
 * Exchange code for access token (Official Account/公众号)
 */
export async function getWeChatMPToken(
  code: string,
  appId: string,
  appSecret: string
): Promise<WeChatTokenResponse> {
  const url = `${WECHAT_MP_URLS.token}?appid=${appId}&secret=${appSecret}&code=${code}&grant_type=authorization_code`

  const response = await $fetch(url)

  // WeChat API returns JSON, but $fetch might return it as string
  // Need to parse it manually if it's a string
  if (typeof response === 'string') {
    return JSON.parse(response)
  }

  return response as WeChatTokenResponse
}

/**
 * Get WeChat user info
 */
export async function getWeChatUserInfo(
  accessToken: string,
  openid: string
): Promise<WeChatUserInfo> {
  const url = `${WECHAT_OPEN_URLS.userinfo}?access_token=${accessToken}&openid=${openid}`

  const response = await $fetch(url)

  // WeChat API returns JSON, but $fetch might return it as string
  // Need to parse it manually if it's a string
  if (typeof response === 'string') {
    return JSON.parse(response)
  }

  return response as WeChatUserInfo
}