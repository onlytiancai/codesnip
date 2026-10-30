/**
 * 加载 server/.env。
 *
 * **为什么自己解析而不用 process.loadEnvFile()**：
 * loadEnvFile 不覆盖 process.env 里已存在的变量 —— 那样「改完 .env 点重新加载」
 * 永远不会生效。util.parseEnv（Node ≥20.12）让我们自己决定覆盖策略。
 *
 * **为什么 .env 优先于 shell 环境变量**：
 * 这个项目是本地单用的，.env 才是「这台机器的真实配置」。
 * 反过来（shell 优先）会让「点重新加载」在有 export 的时候行为不一致。
 * 想临时覆盖某一项，用 `SERVER_ENV_FILE=/path/to/other.env`。
 */

import { existsSync, readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { parseEnv } from 'node:util'

/** 默认的 env 文件：server/.env（src/config/env.ts → ../../ = server/）。 */
export const DEFAULT_ENV_FILE = fileURLToPath(
  new URL('../../.env', import.meta.url),
)

/** 当前生效的 env 文件路径（可用 SERVER_ENV_FILE 覆盖）。 */
export function envFilePath(): string {
  return process.env.SERVER_ENV_FILE || DEFAULT_ENV_FILE
}

/**
 * 读 env 文件并写进 process.env。
 *
 * - 文件不存在 → 当作没有（不抛错）。首次 clone 仓库时 .env 本来就不在 git 里。
 * - 解析出错 → 抛错。`.env` 写坏了应该立刻知道，而不是静默用旧值。
 * - 返回实际写进去的键值对（调用方可以打日志说「加载了哪几个变量」）。
 */
export function loadEnv(file = envFilePath()): Record<string, string> {
  if (!existsSync(file)) return {}

  const parsed = parseEnv(readFileSync(file, 'utf8'))
  const applied: Record<string, string> = {}
  for (const [key, value] of Object.entries(parsed)) {
    if (value === undefined) continue
    process.env[key] = value
    applied[key] = value
  }
  return applied
}

/** 从 process.env 里取一个变量，空串当作没设。 */
export function envValue(name: string): string | undefined {
  const v = process.env[name]
  return v && v.trim() ? v.trim() : undefined
}
