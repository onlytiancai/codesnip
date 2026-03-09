declare module '#auth-utils' {
  interface User {
    id: number
    email: string
    name?: string | null
    avatar?: string | null
    role?: 'USER' | 'ADMIN'
    hasPassword?: boolean
    accounts?: Array<{
      id: number
      provider: string
      providerAccountId: string
      access_token?: string | null
      refresh_token?: string | null
    }>
  }

  interface UserSession {
    user?: User
    createdAt?: string
    loggedInAt?: number
  }

  interface SecureSessionData {
    // Add your own fields if needed
  }
}

export {}