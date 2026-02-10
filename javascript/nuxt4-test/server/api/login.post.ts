export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { username, password } = body

  // 模拟数据库校验
  if (username === 'admin' && password === '123456') {

      await setUserSession(event, {
          // User data
          user: {
              login: 'onlytiancai'
          },
          // Private data accessible only on server/ routes
          secure: {
              apiToken: '1234567890'
          },
          // Any extra fields for the session data
          loggedInAt: new Date()
      })

    return {
      code: 0,
      message: '登录成功'
    }
  }

  return {
    code: 1,
    message: '用户名或密码错误'
  }
})
