/**
 * 请求级的「当前用户」。
 *
 * 用 AsyncLocalStorage 而不是把 userId 一路穿到 service 层每个方法签名里 ——
 * 后者要改 AiService / TranslateService 的公开接口和一堆测试。
 *
 * ALS 是**请求级**的：中间件里 `runAsUser(user, next)` 之后，整条下游链路
 * （含 async handler 里的 await）都跑在这个上下文里，并发请求互不干扰。
 *
 * CLI / 单测不走中间件 → 上下文是空的 → 不记账。
 */

import { AsyncLocalStorage } from 'node:async_hooks'

export interface AuthUser {
  id: number
  username: string
}

const storage = new AsyncLocalStorage<AuthUser>()

/** 在指定用户上下文里跑一段代码（通常就是 next()）。 */
export function runAsUser<T>(user: AuthUser, fn: () => T): T {
  return storage.run(user, fn)
}

/** 当前请求的用户；没有登录（或不在 HTTP 请求里）就是 undefined。 */
export function currentUser(): AuthUser | undefined {
  return storage.getStore()
}
