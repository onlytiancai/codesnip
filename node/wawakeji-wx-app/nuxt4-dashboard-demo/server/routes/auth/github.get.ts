import { findUserByOAuth, createUser, linkOAuthAccount, findUserByEmail } from '../../utils/db-auth'

export default defineOAuthGitHubEventHandler({
  config: {
    emailRequired: true,
  },
  async onSuccess(event, { user }) {
    // First check if OAuth account already exists
    let dbUser = await findUserByOAuth('github', user.id.toString())

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
      return
    }

    // Check if user with same email exists
    dbUser = await findUserByEmail(user.email)

    if (dbUser) {
      // User exists, link GitHub account
      await linkOAuthAccount(
        dbUser.id,
        'github',
        user.id.toString(),
        user.accessToken,
        user.refreshToken
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
      return
    }

    // Create new user and link GitHub account
    const newUser = await createUser({
      email: user.email,
      name: user.name || user.login,
      avatar: user.avatar_url,
    })

    await linkOAuthAccount(
      newUser.id,
      'github',
      user.id.toString(),
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
    console.error('GitHub OAuth error:', error)
  },
})