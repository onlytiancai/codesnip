import { beforeEach, describe, expect, it, vi } from 'vitest'
import type { ProviderConfig } from '../src/config/types.js'

/**
 * 三家客户端的请求形状 + 降级链。
 *
 * SDK 本身用 mock 挡掉，只验证「我们到底往 API 发了什么」。
 */

const mocks = vi.hoisted(() => ({
  ollamaChat: vi.fn(),
  ollamaList: vi.fn(),
  ollamaCtor: vi.fn(),
  openaiCreate: vi.fn(),
  openaiModels: vi.fn(),
  openaiCtor: vi.fn(),
  anthropicCreate: vi.fn(),
  anthropicModels: vi.fn(),
  anthropicCtor: vi.fn(),
}))

vi.mock('ollama', () => ({
  Ollama: class {
    constructor(opts: unknown) {
      mocks.ollamaCtor(opts)
    }
    chat = mocks.ollamaChat
    list = mocks.ollamaList
  },
}))

vi.mock('openai', () => ({
  default: class {
    constructor(opts: unknown) {
      mocks.openaiCtor(opts)
    }
    chat = { completions: { create: mocks.openaiCreate } }
    models = { list: mocks.openaiModels }
  },
}))

vi.mock('@anthropic-ai/sdk', () => ({
  default: class {
    constructor(opts: unknown) {
      mocks.anthropicCtor(opts)
    }
    messages = { create: mocks.anthropicCreate }
    models = { list: mocks.anthropicModels }
  },
}))

const { stripReasoning } = await import('../src/llm/types.js')
const { createOllamaClient } = await import('../src/llm/ollamaClient.js')
const { createOpenAiClient } = await import('../src/llm/openaiClient.js')
const { createAnthropicClient } = await import('../src/llm/anthropicClient.js')

const SCHEMA = {
  type: 'object',
  properties: { text: { type: 'string' } },
  required: ['text'],
}

function provider(over: Partial<ProviderConfig> = {}): ProviderConfig {
  return {
    id: 'p',
    label: 'P',
    kind: 'openai',
    baseUrl: 'http://localhost:1234/v1',
    model: 'm',
    structuredOutput: 'schema',
    ...over,
  }
}

/** 造一个带 status 的错误（SDK 的 APIError 形状）。 */
function httpError(status: number, message: string): Error {
  const err = new Error(message) as Error & { status: number }
  err.status = status
  return err
}

beforeEach(() => {
  // reset 而不是 clear：clear 只清调用记录，mock 的实现会留着，
  // 下一个用例就会拿到上一个用例的返回值。
  vi.resetAllMocks()
})

describe('Ollama 客户端', () => {
  it('schema 模式：把 schema 交给 format 做语法约束', async () => {
    mocks.ollamaChat.mockResolvedValue({
      message: { content: '{"text":"hi"}' },
    })
    const client = createOllamaClient(provider({ kind: 'ollama' }))

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    const arg = mocks.ollamaChat.mock.calls[0][0]
    expect(arg.model).toBe('m')
    expect(arg.format).toEqual(SCHEMA)
    expect(arg.messages).toEqual([
      { role: 'system', content: 'S' },
      { role: 'user', content: 'U' },
    ])
    expect(res.structured).toBe(true)
    expect(res.content).toBe('{"text":"hi"}')
  })

  it('prompt 模式：不传 format，但 system 追加 JSON 契约', async () => {
    mocks.ollamaChat.mockResolvedValue({ message: { content: '{}' } })
    const client = createOllamaClient(
      provider({ kind: 'ollama', structuredOutput: 'prompt' }),
    )

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    const arg = mocks.ollamaChat.mock.calls[0][0]
    expect(arg.format).toBeUndefined()
    expect(arg.messages[0].content).toContain('S')
    expect(arg.messages[0].content).toContain(
      'Respond with a single JSON object',
    )
    expect(res.structured).toBe(false)
  })

  it('温度只在显式配置时才发', async () => {
    mocks.ollamaChat.mockResolvedValue({ message: { content: 'x' } })
    const client = createOllamaClient(provider({ kind: 'ollama' }))
    await client.chat({ system: 'S', user: 'U' })
    expect(mocks.ollamaChat.mock.calls[0][0].options).toBeUndefined()

    mocks.ollamaChat.mockClear()
    const client2 = createOllamaClient(
      provider({ kind: 'ollama', temperature: 0.3 }),
    )
    await client2.chat({ system: 'S', user: 'U' })
    expect(mocks.ollamaChat.mock.calls[0][0].options).toEqual({
      temperature: 0.3,
    })
  })

  it('baseUrl 传给 Ollama 的 host', () => {
    createOllamaClient(
      provider({ kind: 'ollama', baseUrl: 'http://127.0.0.1:11434' }),
    )
    expect(mocks.ollamaCtor).toHaveBeenCalledWith({
      host: 'http://127.0.0.1:11434',
    })
  })

  it('listModels 取 models[].name', async () => {
    mocks.ollamaList.mockResolvedValue({ models: [{ name: 'a' }, { name: 'b' }] })
    const client = createOllamaClient(provider({ kind: 'ollama' }))
    expect(await client.listModels()).toEqual(['a', 'b'])
  })
})

describe('OpenAI 客户端', () => {
  it('schema 模式：response_format.json_schema 且 strict=false', async () => {
    mocks.openaiCreate.mockResolvedValue({
      choices: [{ message: { content: '{"text":"hi"}' } }],
    })
    const client = createOpenAiClient(provider())

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    const arg = mocks.openaiCreate.mock.calls[0][0]
    expect(arg.response_format.type).toBe('json_schema')
    expect(arg.response_format.json_schema.name).toBe('result')
    expect(arg.response_format.json_schema.schema).toEqual(SCHEMA)
    expect(arg.response_format.json_schema.strict).toBe(false)
    expect(arg.messages[0]).toEqual({ role: 'system', content: 'S' })
    expect(res.structured).toBe(true)
  })

  it('端点不认 json_schema（400）→ 降级 json_object', async () => {
    mocks.openaiCreate
      .mockRejectedValueOnce(
        httpError(400, "Unsupported parameter: 'response_format'"),
      )
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"text":"hi"}' } }],
      })
    const client = createOpenAiClient(provider())

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    expect(mocks.openaiCreate).toHaveBeenCalledTimes(2)
    const second = mocks.openaiCreate.mock.calls[1][0]
    expect(second.response_format.type).toBe('json_object')
    expect(res.structured).toBe(true)
  })

  it('连 json_object 都不认 → 降到纯提示词（system 里带 JSON 契约）', async () => {
    mocks.openaiCreate
      .mockRejectedValueOnce(httpError(400, 'no response_format'))
      .mockRejectedValueOnce(httpError(400, 'no response_format'))
      .mockResolvedValueOnce({
        choices: [{ message: { content: '{"text":"hi"}' } }],
      })
    const client = createOpenAiClient(provider())

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    expect(mocks.openaiCreate).toHaveBeenCalledTimes(3)
    const third = mocks.openaiCreate.mock.calls[2][0]
    expect(third.response_format).toBeUndefined()
    expect(third.messages[0].content).toContain(
      'Respond with a single JSON object',
    )
    expect(res.structured).toBe(false)
  })

  it('非 400 的错误不降级，直接抛', async () => {
    mocks.openaiCreate.mockRejectedValue(httpError(401, 'bad key'))
    const client = createOpenAiClient(provider())

    await expect(
      client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA }),
    ).rejects.toThrow(/bad key/)
    expect(mocks.openaiCreate).toHaveBeenCalledTimes(1)
  })

  it('max_tokensParam 决定用哪个字段', async () => {
    mocks.openaiCreate.mockResolvedValue({
      choices: [{ message: { content: 'x' } }],
    })
    const client = createOpenAiClient(
      provider({ maxTokens: 512, maxTokensParam: 'max_completion_tokens' }),
    )
    await client.chat({ system: 'S', user: 'U' })
    const arg = mocks.openaiCreate.mock.calls[0][0]
    expect(arg.max_completion_tokens).toBe(512)
    expect(arg.max_tokens).toBeUndefined()
  })

  it('默认用 max_tokens（兼容面最广）', async () => {
    mocks.openaiCreate.mockResolvedValue({
      choices: [{ message: { content: 'x' } }],
    })
    const client = createOpenAiClient(provider({ maxTokens: 512 }))
    await client.chat({ system: 'S', user: 'U' })
    expect(mocks.openaiCreate.mock.calls[0][0].max_tokens).toBe(512)
  })

  it('温度只在显式配置时才发', async () => {
    mocks.openaiCreate.mockResolvedValue({
      choices: [{ message: { content: 'x' } }],
    })
    const client = createOpenAiClient(provider())
    await client.chat({ system: 'S', user: 'U' })
    expect(mocks.openaiCreate.mock.calls[0][0].temperature).toBeUndefined()
  })

  it('apiKey / baseURL 传给 SDK', () => {
    createOpenAiClient(provider({ baseUrl: 'http://x/v1' }), 'sk-123')
    expect(mocks.openaiCtor).toHaveBeenCalledWith(
      expect.objectContaining({ apiKey: 'sk-123', baseURL: 'http://x/v1' }),
    )
  })

  it('listModels 取 data[].id', async () => {
    mocks.openaiModels.mockResolvedValue({ data: [{ id: 'gpt-x' }] })
    const client = createOpenAiClient(provider())
    expect(await client.listModels()).toEqual(['gpt-x'])
  })
})

describe('Anthropic 客户端', () => {
  it('schema 模式：用 tool-use 强制 JSON，tool_choice 锁定', async () => {
    mocks.anthropicCreate.mockResolvedValue({
      content: [{ type: 'tool_use', id: 't1', name: 'result', input: { text: 'hi' } }],
    })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    const arg = mocks.anthropicCreate.mock.calls[0][0]
    expect(arg.system).toBe('S')
    expect(arg.messages).toEqual([{ role: 'user', content: 'U' }])
    expect(arg.tools[0].name).toBe('result')
    expect(arg.tools[0].input_schema).toEqual(SCHEMA)
    expect(arg.tool_choice).toEqual({ type: 'tool', name: 'result' })
    expect(res.content).toBe('{"text":"hi"}')
    expect(res.structured).toBe(true)
  })

  it('没有 tool_use block 时退回拼 text block', async () => {
    mocks.anthropicCreate.mockResolvedValue({
      content: [{ type: 'text', text: 'plain answer' }],
    })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })
    expect(res.content).toBe('plain answer')
  })

  it('端点不支持 tools（400）→ 降级成纯提示词', async () => {
    mocks.anthropicCreate
      .mockRejectedValueOnce(httpError(400, 'tools not supported'))
      .mockResolvedValueOnce({
        content: [{ type: 'text', text: '{"text":"hi"}' }],
      })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))

    const res = await client.chat({ system: 'S', user: 'U', jsonSchema: SCHEMA })

    expect(mocks.anthropicCreate).toHaveBeenCalledTimes(2)
    const second = mocks.anthropicCreate.mock.calls[1][0]
    expect(second.tools).toBeUndefined()
    expect(second.tool_choice).toBeUndefined()
    expect(second.system).toContain('Respond with a single JSON object')
    expect(res.structured).toBe(false)
  })

  it('max_tokens 必填，默认 16000', async () => {
    mocks.anthropicCreate.mockResolvedValue({ content: [] })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))
    await client.chat({ system: 'S', user: 'U' })
    expect(mocks.anthropicCreate.mock.calls[0][0].max_tokens).toBe(16000)
  })

  it('不发 temperature —— Anthropic API 已移除采样参数', async () => {
    mocks.anthropicCreate.mockResolvedValue({ content: [] })
    const client = createAnthropicClient(
      provider({ kind: 'anthropic', temperature: 0.5 }),
    )
    await client.chat({ system: 'S', user: 'U' })
    const arg = mocks.anthropicCreate.mock.calls[0][0]
    expect('temperature' in arg).toBe(false)
  })

  it('apiKey / baseURL 传给 SDK', () => {
    createAnthropicClient(
      provider({ kind: 'anthropic', baseUrl: 'http://x' }),
      'sk-ant-123',
    )
    expect(mocks.anthropicCtor).toHaveBeenCalledWith(
      expect.objectContaining({ apiKey: 'sk-ant-123', baseURL: 'http://x' }),
    )
  })

  it('listModels 取 data[].id', async () => {
    mocks.anthropicModels.mockResolvedValue({ data: [{ id: 'claude-x' }] })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))
    expect(await client.listModels()).toEqual(['claude-x'])
  })
})


describe('stripReasoning', () => {
  it('没有 think 块时原样返回（语法检查的「原文透传」语义不受影响）', () => {
    expect(stripReasoning('plain answer')).toBe('plain answer')
    expect(stripReasoning('  带空格的答案  ')).toBe('  带空格的答案  ')
  })

  it('剥掉完整的 <think>…</think> 块', () => {
    expect(stripReasoning('<think>\nreasoning here\n</think>\n\nOK')).toBe('OK')
  })

  it('剥掉没闭合的 <think> 块（后面全是思考）', () => {
    expect(stripReasoning('<think>\nstill reasoning...')).toBe('')
  })

  it('剥掉多个 think 块', () => {
    expect(stripReasoning('<think>a</think>X<think>b</think>Y')).toBe('XY')
  })

  it('OpenAI 客户端会把 content 里的 think 块剥掉', async () => {
    mocks.openaiCreate.mockResolvedValue({
      choices: [
        {
          message: {
            content: '<think>\nlet me think\n</think>\n\nThe answer is 42.',
          },
        },
      ],
    })
    const client = createOpenAiClient(provider())
    const res = await client.chat({ system: 'S', user: 'U' })
    expect(res.content).toBe('The answer is 42.')
  })

  it('Anthropic 客户端会剥掉 text block 里的 think 块', async () => {
    mocks.anthropicCreate.mockResolvedValue({
      content: [{ type: 'text', text: '<think>hmm</think>final answer' }],
    })
    const client = createAnthropicClient(provider({ kind: 'anthropic' }))
    const res = await client.chat({ system: 'S', user: 'U' })
    expect(res.content).toBe('final answer')
  })
})
