import { describe, expect, it, vi } from 'vitest'
import { createTranslateService } from '../src/services/translate.js'
import { createOllamaService } from '../src/services/ollama.js'
import type { Ollama } from 'ollama'

/**
 * 服务层测试：验证 prompt 真的按预期组装进了 Ollama 的 messages，
 * 且响应被正确解析。这是 mock 路由层测不到的一层。
 */

/** 构造一个只关心「收到什么 messages」的 ollama 桩。 */
function stubClient(reply: string) {
  const chat = vi.fn(async () => ({ message: { role: 'assistant', content: reply } }))
  return { client: { chat } as unknown as Ollama, chat }
}

function systemOf(chat: ReturnType<typeof vi.fn>): string {
  const messages = chat.mock.calls[0][0].messages
  return messages.find((m: { role: string }) => m.role === 'system').content
}

function userOf(chat: ReturnType<typeof vi.fn>): string {
  const messages = chat.mock.calls[0][0].messages
  return messages.find((m: { role: string }) => m.role === 'user').content
}

describe('createTranslateService', () => {
  it('无 tone 时 system 就是基础 prompt', async () => {
    const { client, chat } = stubClient('I went to the park yesterday.')
    const svc = createTranslateService({ client, model: 'm' })

    const r = await svc.translate('我昨天去公园。')

    expect(r.translation).toBe('I went to the park yesterday.')
    expect(systemOf(chat)).not.toMatch(/tone/)
  })

  it('带 tone 时 system 追加 tone 指令', async () => {
    const { client, chat } = stubClient('Formal prose.')
    const svc = createTranslateService({ client, model: 'm' })

    await svc.translate('你好', { tone: 'formal' })

    expect(systemOf(chat)).toContain('Write in a formal tone')
    expect(userOf(chat)).toContain('Chinese: 你好')
  })

  it('空文本直接抛错，不打模型', async () => {
    const { client, chat } = stubClient('x')
    const svc = createTranslateService({ client, model: 'm' })

    await expect(svc.translate('  ')).rejects.toThrow(/non-empty/)
    expect(chat).not.toHaveBeenCalled()
  })
})

describe('createOllamaService', () => {
  it('rewriteSelection：把 kind 带进 system，返回候选', async () => {
    const { client, chat } = stubClient('["essential","of great importance"]')
    const svc = createOllamaService({ client, model: 'm' })

    const cands = await svc.rewriteSelection('very important', 'phrase', {
      context: 'Education is very important for children.',
    })

    expect(cands).toEqual(['essential', 'of great importance'])
    expect(systemOf(chat)).toContain('multi-word phrases')
    expect(userOf(chat)).toContain('Selection: very important')
    expect(userOf(chat)).toContain('Context: Education is very important')
  })

  it('rewriteSelection：遵守 maxCandidates', async () => {
    const { client } = stubClient('["a","b","c"]')
    const svc = createOllamaService({ client, model: 'm', defaultMaxCandidates: 2 })

    expect(await svc.rewriteSelection('x', 'word')).toEqual(['a', 'b'])
  })

  it('rewriteSentence：operation + tone 进 system，原句进 user', async () => {
    const { client, chat } = stubClient('Education matters greatly.')
    const svc = createOllamaService({ client, model: 'm' })

    const out = await svc.rewriteSentence('Education matters.', {
      operation: 'rewrite',
      tone: 'persuasive',
    })

    expect(out).toBe('Education matters greatly.')
    expect(systemOf(chat)).toContain('Operation: rewrite')
    expect(systemOf(chat)).toContain('persuasive')
    expect(userOf(chat)).toContain('Sentence: Education matters.')
  })

  it('grammarCheck：模型原文原样返回，不解析', async () => {
    const raw = '「go」应该改成「goes」，因为主语 He 是第三人称单数。'
    const { client, chat } = stubClient(raw)
    const svc = createOllamaService({ client, model: 'm' })

    expect(await svc.grammarCheck('He go to school.')).toBe(raw)
    expect(systemOf(chat)).toContain('English proofreader')
    // 原句直接进 user message，不加任何前缀包装
    expect(userOf(chat)).toBe('He go to school.')
  })

  it('rewriteInstruction：指令原样进 user message', async () => {
    const { client, chat } = stubClient('A more academic rendering.')
    const svc = createOllamaService({ client, model: 'm' })

    const out = await svc.rewriteInstruction('It is very important.', {
      instruction: 'Make it more academic.',
    })

    expect(out).toBe('A more academic rendering.')
    expect(userOf(chat)).toContain('Instruction: Make it more academic.')
  })

  it('空输入在四个方法上都直接抛错，不打模型', async () => {
    const { client, chat } = stubClient('x')
    const svc = createOllamaService({ client, model: 'm' })

    await expect(svc.rewriteSelection(' ', 'word')).rejects.toThrow(/non-empty/)
    await expect(
      svc.rewriteSentence('', { operation: 'rewrite' }),
    ).rejects.toThrow(/non-empty/)
    await expect(svc.grammarCheck(' ')).rejects.toThrow(/non-empty/)
    await expect(
      svc.rewriteInstruction(' ', { instruction: 'x' }),
    ).rejects.toThrow(/non-empty/)

    expect(chat).not.toHaveBeenCalled()
  })
})
