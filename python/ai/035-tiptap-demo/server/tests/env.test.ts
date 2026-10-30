import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import { loadEnv } from '../src/config/env.js'

/**
 * .env 加载：读得到、覆盖得到、文件缺失不炸。
 */

describe('loadEnv', () => {
  let dir: string | undefined
  const touched: string[] = []

  afterEach(() => {
    for (const key of touched.splice(0)) delete process.env[key]
    if (dir) {
      rmSync(dir, { recursive: true, force: true })
      dir = undefined
    }
  })

  function envFile(content: string): string {
    dir = mkdtempSync(join(tmpdir(), 'tiptap-env-'))
    const path = join(dir, '.env')
    writeFileSync(path, content, 'utf8')
    return path
  }

  it('把 .env 里的键写进 process.env', () => {
    const path = envFile('FOO=bar\nBAZ=qux\n')
    const applied = loadEnv(path)

    expect(applied).toEqual({ FOO: 'bar', BAZ: 'qux' })
    expect(process.env.FOO).toBe('bar')
    expect(process.env.BAZ).toBe('qux')
    touched.push('FOO', 'BAZ')
  })

  it('覆盖已存在的同名变量（否则「改完 .env 重新加载」不生效）', () => {
    process.env.FOO = 'old'
    touched.push('FOO')

    const path = envFile('FOO=new\n')
    loadEnv(path)

    expect(process.env.FOO).toBe('new')
  })

  it('引号和注释都处理掉', () => {
    const path = envFile('# 这是注释\nQUOTED="hello world"\n')
    const applied = loadEnv(path)

    expect(applied.QUOTED).toBe('hello world')
    expect(process.env.QUOTED).toBe('hello world')
    touched.push('QUOTED')
  })

  it('文件不存在 → 当作没有，不抛错', () => {
    const applied = loadEnv('/definitely/not/here/.env')
    expect(applied).toEqual({})
  })
})
