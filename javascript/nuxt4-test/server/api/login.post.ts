export default defineEventHandler(async (event) => {
  const body = await readBody(event)
  const { username, password } = body

  // const hashedPassword = await hashPassword('123456')
  // console.log('hashedPassword:', hashedPassword)
  const hashedPassword = '$scrypt$n=16384,r=8,p=1$gAJArRUrtpKouqXJ6wG5Pw$r+c7hYDClxrTuCyCRyLU3VuvkMc+uwjf+xjhKXKqGusvUzxOxdCGORkk8P+36d8OVsqxWsWJdaLIXm1bgJyiCw';

  // 模拟数据库校验
  if (username === 'admin' && await verifyPassword(hashedPassword, password)) {

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
