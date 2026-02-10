export default defineOAuthWeixinEventHandler({
  config: {},
  async onSuccess(event, { user, tokens }) {

    await setUserSession(event, {
      user  : user,
      tokens: tokens
    })
    return sendRedirect(event, '/')
  },
  // Optional, will return a json error and 401 status code by default
  onError(event, error) {
    console.error('Weixin OAuth error:', error)
    return sendRedirect(event, '/')
  },
})