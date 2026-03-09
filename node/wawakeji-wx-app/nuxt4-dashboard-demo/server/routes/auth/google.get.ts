import { findUserByOAuth, createUser, linkOAuthAccount, findUserByEmail } from '../../utils/db-auth'

export default defineOAuthGoogleEventHandler({
  config: {
    scope: ['openid', 'email', 'profile'],
  },
  async onSuccess(event, { user }) {
    // First check if OAuth account already exists
    let dbUser = await findUserByOAuth('google', user.sub)

    if (dbUser) {
      // Already exists, login directly
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.picture,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Check if user with same email exists
    dbUser = await findUserByEmail(user.email)

    if (dbUser) {
      // User exists, link Google account
      await linkOAuthAccount(
        dbUser.id,
        'google',
        user.sub,
        user.accessToken,
        user.refreshToken
      )
      await setUserSession(event, {
        user: {
          id: dbUser.id,
          email: dbUser.email,
          name: dbUser.name,
          avatar: dbUser.picture,
          role: dbUser.role,
          hasPassword: !!dbUser.password,
        },
      })
      return
    }

    // Create new user and link Google account
    const newUser = await createUser({
      email: user.email,
      name: user.name,
      avatar: user.picture,
    })

    await linkOAuthAccount(
      newUser.id,
      'google',
      user.sub,
      user.accessToken,
      user.refreshToken
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
  },
  onError(event, error) {
    console.error('Google OAuth error:', error)
  },
})