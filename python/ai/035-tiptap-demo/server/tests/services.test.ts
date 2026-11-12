import { describe, expect, it, vi } from 'vitest'
import { buildServices, stubLlmClient } from './helpers.js'
import type { ChatRequest } from '../src/llm/types.js'

/**
 * 服务层测试：验证提示词真的按预期组装进了请求，且响应被正确解析。
 * 这是 mock 路由层测不到的一层。
 */

function systemOf(chat: ReturnType<typeof vi.fn>): string {
  return chat.mock.calls[0][0].system as string
}

function userOf(chat: ReturnType<typeof vi.fn>): string {
  return chat.mock.calls[0][0].user as string
}

function requestOf(chat: ReturnType<typeof vi.fn>): ChatRequest {
  return chat.mock.calls[0][0] as ChatRequest
}

describe('createTranslateService', () => {
  it('无 tone 时 system 就是基础 prompt', async () => {
    const { client, chat } = stubLlmClient('I went to the park yesterday.')
    const { translate } = buildServices(client)

    const r = await translate.translate('我昨天去公园。')

    expect(r.translation).toBe('I went to the park yesterday.')
    expect(systemOf(chat)).not.toMatch(/tone/)
  })

  it('带 tone 时 system 追加 tone 指令', async () => {
    const { client, chat } = stubLlmClient('Formal prose.')
    const { translate } = buildServices(client)

    await translate.translate('你好', { tone: 'formal' })

    expect(systemOf(chat)).toContain('Write in a formal tone')
    expect(userOf(chat)).toContain('Chinese: 你好')
  })

  it('空文本直接抛错，不打模型', async () => {
    const { client, chat } = stubLlmClient('x')
    const { translate } = buildServices(client)

    await expect(translate.translate('  ')).rejects.toThrow(/non-empty/)
    expect(chat).not.toHaveBeenCalled()
  })
})

describe('createAiService', () => {
  it('rewriteSelection：把 kind 带进 system，返回候选', async () => {
    const { client, chat } = stubLlmClient('["essential","of great importance"]')
    const { ai } = buildServices(client)

    const cands = await ai.rewriteSelection('very important', 'phrase', {
      context: 'Education is very important for children.',
    })

    expect(cands).toEqual(['essential', 'of great importance'])
    expect(systemOf(chat)).toContain('multi-word phrases')
    expect(userOf(chat)).toContain('Selection: very important')
    expect(userOf(chat)).toContain('Context: Education is very important')
  })

  it('rewriteSelection：遵守 maxCandidates', async () => {
    const { client } = stubLlmClient('["a","b","c"]')
    const { ai } = buildServices(client)

    expect(await ai.rewriteSelection('x', 'word')).toEqual(['a', 'b', 'c'])
    expect(
      await ai.rewriteSelection('x', 'word', { maxCandidates: 2 }),
    ).toEqual(['a', 'b'])
  })

  it('rewriteSentence：operation + tone 进 system，原句进 user', async () => {
    const { client, chat } = stubLlmClient('Education matters greatly.')
    const { ai } = buildServices(client)

    const out = await ai.rewriteSentence('Education matters.', {
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
    const { client, chat } = stubLlmClient(raw)
    const { ai } = buildServices(client)

    expect(await ai.grammarCheck('He go to school.')).toBe(raw)
    expect(systemOf(chat)).toContain('English proofreader')
    // 原句直接进 user message，不加任何前缀包装
    expect(userOf(chat)).toBe('He go to school.')
  })

  it('rewriteInstruction：指令原样进 user message', async () => {
    const { client, chat } = stubLlmClient('A more academic rendering.')
    const { ai } = buildServices(client)

    const out = await ai.rewriteInstruction('It is very important.', {
      instruction: 'Make it more academic.',
    })

    expect(out).toBe('A more academic rendering.')
    expect(userOf(chat)).toContain('Instruction: Make it more academic.')
  })

  it('空输入在四个方法上都直接抛错，不打模型', async () => {
    const { client, chat } = stubLlmClient('x')
    const { ai } = buildServices(client)

    await expect(ai.rewriteSelection(' ', 'word')).rejects.toThrow(/non-empty/)
    await expect(
      ai.rewriteSentence('', { operation: 'rewrite' }),
    ).rejects.toThrow(/non-empty/)
    await expect(ai.grammarCheck(' ')).rejects.toThrow(/non-empty/)
    await expect(
      ai.rewriteInstruction(' ', { instruction: 'x' }),
    ).rejects.toThrow(/non-empty/)

    expect(chat).not.toHaveBeenCalled()
  })

  it('text 模式的模板不会带 jsonSchema', async () => {
    const { client, chat } = stubLlmClient('ok')
    const { ai } = buildServices(client)

    await ai.grammarCheck('He go to school.')

    expect(requestOf(chat).jsonSchema).toBeUndefined()
  })
})
