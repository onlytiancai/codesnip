import { createApp } from './app.js'
import { envValue, loadEnv } from './config/env.js'
import { DEFAULT_DB_PATH, openDatabase } from './config/db.js'
import { ConfigStore } from './config/store.js'
import { UserStore } from './config/userStore.js'
import { ClientRegistry } from './llm/registry.js'
import { createActiveProviderGetter } from './services/activeProvider.js'
import { createAiService } from './services/ai.js'
import { createTranslateService } from './services/translate.js'
import { createDictionaryService } from './services/dictionary.js'
import { currentUser } from './services/requestContext.js'
import type { TokenUsage } from './llm/types.js'
import { debugLog } from './utils/debug.js'

// 必须最先跑：PORT / OLLAMA_* / API key 都从 server/.env 里读，
// 而 defaultProvider()（在 new ConfigStore() 里）会读 OLLAMA_HOST。
loadEnv()

const PORT = Number(process.env.PORT ?? 8787)

// 一个库文件装下所有东西：配置 + 用户 + 会话 + token 用量
const dbPath = envValue('DB_PATH') ?? DEFAULT_DB_PATH
const db = openDatabase(dbPath)
debugLog('[db]', dbPath)

const configStore = new ConfigStore({ db })
const registry = new ClientRegistry()
const getActive = createActiveProviderGetter(configStore, registry)
const userStore = new UserStore({ db })

// 第一次启动：播种一个默认账号（密码在 server/.env 的 ADMIN_PASSWORD 里）
if (userStore.countUsers() === 0) {
  const username = envValue('ADMIN_USERNAME') ?? 'admin'
  const password = envValue('ADMIN_PASSWORD') ?? 'admin123'
  userStore.createUser(username, password, { role: 'admin' })
  // **不回显密码** —— 它在 server/.env 里，日志里只说账号名
  console.log(
    `[tiptap-rewrite-server] 已创建默认账号：${username}（密码见 server/.env 的 ADMIN_PASSWORD）`,
  )
}

/**
 * 管理员名单。
 *
 * ADMIN_USERNAME（主管理员）+ ADMIN_USERS（逗号分隔，可以多个）。
 * 名单里的人每次启动都会被提成 admin —— .env 就是管理员名单的唯一来源。
 */
const adminNames = new Set<string>()
const primaryAdmin = envValue('ADMIN_USERNAME')
if (primaryAdmin) adminNames.add(primaryAdmin)
for (const raw of (envValue('ADMIN_USERS') ?? '').split(',')) {
  const name = raw.trim()
  if (name) adminNames.add(name)
}
for (const name of adminNames) {
  const user = userStore.findUserByUsername(name)
  if (user && user.role !== 'admin') {
    userStore.setUserRole(user.id, 'admin')
    console.log(`[tiptap-rewrite-server] 已把 ${name} 设为管理员`)
  }
}

/**
 * 记一次 token 消耗。
 *
 * 「记到谁头上」从 AsyncLocalStorage 里取当前登录用户 —— 登录守卫挂上去的。
 * 本地工具调用（curl / CLI）没有用户 → 不记。
 */
function recordUsage(usage: TokenUsage): void {
  const user = currentUser()
  if (!user) return
  userStore.addUsage(user.id, usage.inputTokens, usage.outputTokens)
  debugLog(
    '[usage]',
    user.username,
    `+${usage.inputTokens} in / +${usage.outputTokens} out`,
  )
}

const aiService = createAiService({ getActive, recordUsage })
const translateService = createTranslateService({ getActive, recordUsage })
const dictionaryService = createDictionaryService()
const app = createApp({
  aiService,
  translateService,
  configStore,
  registry,
  dictionaryService,
  userStore,
  recordUsage,
})

app.listen(PORT, () => {
  const provider = configStore.getActiveProvider()
  // eslint-disable-next-line no-console
  console.log(`[tiptap-rewrite-server] listening on http://127.0.0.1:${PORT}`)
  // eslint-disable-next-line no-console
  console.log(
    `[tiptap-rewrite-server] provider: ${provider.label} (${provider.kind} / ${provider.model})`,
  )
  // eslint-disable-next-line no-console
  console.log(`[tiptap-rewrite-server] db: ${dbPath}`)
})
