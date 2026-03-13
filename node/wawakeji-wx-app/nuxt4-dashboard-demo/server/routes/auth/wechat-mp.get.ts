import {
  findUserByOAuth,
  createUser,
  linkOAuthAccount,
  findUserByEmail,
} from '../../utils/db-auth'
import {
  getWeChatMPToken,
  getWeChatUserInfo,
  WECHAT_MP_URLS,
  type WeChatTokenResponse,
} from '../../utils/wechat'

export default defineEventHandler(async (event) => {
  const config = useRuntimeConfig()
  const query = getQuery(event)

  const appId = config.oauth?.wechatMp?.appid as string
  const appSecret = config.oauth?.wechatMp?.secret as string

  if (!appId || !appSecret) {
    throw createError({
      statusCode: 500,
      message: 'WeChat MP OAuth is not configured',
    })
  }

  // Use configured base URL or fall back to request origin
  const baseUrl = config.public.baseUrl || getRequestURL(event).origin
  const redirectUri = `${baseUrl}/auth/wechat-mp`

  // Handle callback from WeChat
  if (query.code) {
    try {
      // Exchange code for access token
      const tokenResponse: WeChatTokenResponse = await getWeChatMPToken(
        query.code as string,
        appId,
        appSecret
      )

      if (tokenResponse.errcode) {
        console.error('[WeChat MP OAuth] Token error:', tokenResponse.errcode, tokenResponse.errmsg)
        throw new Error(`WeChat MP OAuth error: ${tokenResponse.errcode} - ${tokenResponse.errmsg}`)
      }

      const openid = tokenResponse.openid
      const unionid = tokenResponse.unionid
      const providerAccountId = unionid || openid

      if (!providerAccountId) {
        throw new Error('WeChat MP OAuth: No openid or unionid received')
      }

      // Try to get additional user info (nickname, avatar)
      let nickname: string | undefined
      let headimgurl: string | undefined

      // Try to get user info if scope is snsapi_userinfo
      if (tokenResponse.scope === 'snsapi_userinfo') {
        try {
          const userInfo = await getWeChatUserInfo(
            tokenResponse.access_token,
            openid
          )

          if (!userInfo.errcode) {
            nickname = userInfo.nickname
            headimgurl = userInfo.headimgurl
          }
        } catch (e) {
          // User info fetch failed, continue without nickname/avatar
        }
      }

      // Check if OAuth account already exists
      let dbUser = await findUserByOAuth('wechat-mp', providerAccountId)

      if (dbUser) {
        // Already exists, login directly
        await setUserSession(event, {
          user: {
            id: dbUser.id,
            email: dbUser.email,
            name: dbUser.name,
            avatar: dbUser.avatar,
            role: dbUser.role,
            hasPassword: !!dbUser.password,
          },
        })
        return sendRedirect(event, '/')
      }

      // Create email from openid (WeChat doesn't always provide email)
      const email = `wechat_mp_${openid}@wechat.placeholder`

      // Check if user with same email exists
      dbUser = await findUserByEmail(email)

      if (dbUser) {
        // User exists, link WeChat MP account
        await linkOAuthAccount(
          dbUser.id,
          'wechat-mp',
          providerAccountId,
          tokenResponse.access_token,
          tokenResponse.refresh_token
        )
        await setUserSession(event, {
          user: {
            id: dbUser.id,
            email: dbUser.email,
            name: dbUser.name,
            avatar: dbUser.avatar,
            role: dbUser.role,
            hasPassword: !!dbUser.password,
          },
        })
        return sendRedirect(event, '/')
      }

      // Create new user and link WeChat MP account
      const newUser = await createUser({
        email,
        name: nickname || `WeChat User ${openid.slice(-6)}`,
        avatar: headimgurl,
      })

      await linkOAuthAccount(
        newUser.id,
        'wechat-mp',
        providerAccountId,
        tokenResponse.access_token,
        tokenResponse.refresh_token
      )

      await setUserSession(event, {
        user: {
          id: newUser.id,
          email: newUser.email,
          name: newUser.name,
          avatar: newUser.avatar,
          role: newUser.role,
          hasPassword: !!newUser.password,
        },
      })

      return sendRedirect(event, '/')
    } catch (error) {
      console.error('[WeChat MP OAuth] Error:', error)
      throw createError({
        statusCode: 400,
        message: 'WeChat MP login failed',
      })
    }
  }

  // Redirect to WeChat OAuth authorize page
  const state = Math.random().toString(36).substring(7)
  // Use snsapi_userinfo to get user's nickname and avatar
  const authUrl = `${WECHAT_MP_URLS.authorize}?appid=${appId}&redirect_uri=${encodeURIComponent(redirectUri)}&response_type=code&scope=snsapi_userinfo&state=${state}#wechat_redirect`

  return sendRedirect(event, authUrl)
})