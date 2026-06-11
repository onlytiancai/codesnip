interface WeixinTokens {
  access_token: string
  expires_in: number
  refresh_token: string
  openid: string
  scope: string
  unionid?: string
}

export default defineEventHandler(async (event) => {
  const session = await getUserSession(event)
  const tokens  = session.tokens as WeixinTokens | undefined
  console.log("tokens?.refresh_token", tokens?.refresh_token)

  if (!tokens?.refresh_token) {
    throw createError({
      statusCode : 401,
      message    : 'No refresh token available'
    })
  }

  const config = useRuntimeConfig(event)

  // Request new tokens from Weixin
  const newTokens =  await $fetch<WeixinTokens>('https://api.weixin.qq.com/sns/oauth2/refresh_token', {
    parseResponse: (txt) => JSON.parse(txt),
    method: 'POST',
    body: new URLSearchParams({
      appid         : config.oauth.weixin.clientId,
      grant_type    : 'refresh_token',
      refresh_token : tokens.refresh_token,
    })
  })

  // Update session with new tokens
  await setUserSession(event, {
    user       : session.user,
    tokens     : newTokens,
    loggedInAt : session.loggedInAt
  })

  return { success: true }
})