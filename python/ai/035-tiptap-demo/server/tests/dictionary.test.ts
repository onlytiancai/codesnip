import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, describe, expect, it } from 'vitest'
import request from 'supertest'
import { createApp } from '../src/app.js'
import { createDictionaryService } from '../src/services/dictionary.js'
import { buildAiService, buildTranslateService } from './helpers.js'

/**
 * 查词：索引 + 查询 + 找不到 + 路由层。
 *
 * 用一个小的 fixture JSONL（格式和 024-cet4 的 vocab_all.json 一样），
 * 不依赖真实的 12MB 词库文件。
 */

function vocabRow(wordHead: string, tranCn: string, sContent: string, sCn: string) {
  return JSON.stringify({
    wordRank: 1,
    headWord: wordHead,
    content: {
      word: {
        wordHead,
        wordId: `CET4_2_${wordHead}`,
        content: {
          usphone: 'ˈtest',
          ukphone: 'ˈtest',
          trans: [{ tranCn, descOther: '英释', descCn: '中释', pos: 'v', tranOther: 'english def' }],
          sentence: { sentences: [{ sContent, sCn }], desc: '例句' },
          phrase: { phrases: [{ pContent: 'test phrase', pCn: '测试短语' }], desc: '短语' },
          syno: { synos: [{ pos: 'v', tran: '同义', hwds: [{ w: 'trial' }] }], desc: '同近' },
          anto: { anto: [{ hwd: 'fail' }], desc: '反义' },
          remMethod: { val: '记忆法', desc: '记忆' },
        },
      },
    },
    bookId: 'CET4_2',
  })
}

describe('查词服务', () => {
  let dir: string | undefined

  afterEach(() => {
    if (dir) {
      rmSync(dir, { recursive: true, force: true })
      dir = undefined
    }
  })

  function fixture(): string {
    dir = mkdtempSync(join(tmpdir(), 'tiptap-dict-'))
    const path = join(dir, 'vocab.json')
    writeFileSync(
      path,
      [
        vocabRow('abandon', '丢弃；放弃', 'How could she abandon her own child?', '她怎么能抛弃自己的孩子？'),
        vocabRow('important', '重要的；重大的', 'This is very important.', '这非常重要。'),
        '{ 这行是坏 JSON，应该被跳过',
        '',
      ].join('\n'),
      'utf8',
    )
    return path
  }

  it('建索引 + 查到词条', async () => {
    const service = createDictionaryService({ path: fixture() })
    // 索引是异步建的，等它一会儿
    await service.reload()

    const entry = await service.lookup('abandon')
    expect(entry).toBeDefined()
    expect(entry?.word).toBe('abandon')
    expect(entry?.trans[0]?.cn).toBe('丢弃；放弃')
    expect(entry?.trans[0]?.pos).toBe('v')
    expect(entry?.sentences[0]?.en).toBe('How could she abandon her own child?')
    expect(entry?.phrases[0]?.cn).toBe('测试短语')
    expect(entry?.antonyms).toEqual(['fail'])
    expect(entry?.mnemonic).toBe('记忆法')
  })

  it('大小写不敏感', async () => {
    const service = createDictionaryService({ path: fixture() })
    await service.reload()
    expect((await service.lookup('IMPORTANT'))?.word).toBe('important')
  })

  it('查不到 → undefined（不是抛错）', async () => {
    const service = createDictionaryService({ path: fixture() })
    await service.reload()
    expect(await service.lookup('zzzqqq')).toBeUndefined()
  })

  it('词库只收词头 → 变形能还原（important → importanc 不行，abandons → abandon）', async () => {
    const service = createDictionaryService({ path: fixture() })
    await service.reload()
    expect((await service.lookup('abandons'))?.word).toBe('abandon')
    expect((await service.lookup('important'))?.word).toBe('important')
  })

  it('坏行被跳过，剩下的词照样能查到', async () => {
    const service = createDictionaryService({ path: fixture() })
    await service.reload()
    expect((await service.lookup('important'))?.word).toBe('important')
  })

  it('词库文件不存在 → status 里有 error，lookup 返回 undefined', async () => {
    const service = createDictionaryService({ path: '/definitely/not/here.json' })
    const status = await service.reload()
    expect(status.loaded).toBe(false)
    expect(status.error).toMatch(/不存在/)
    expect(await service.lookup('abandon')).toBeUndefined()
  })

  it('status 报告词数', async () => {
    const service = createDictionaryService({ path: fixture() })
    const status = await service.reload()
    expect(status.loaded).toBe(true)
    expect(status.count).toBe(2)
  })
})

describe('查词路由', () => {
  let dir: string | undefined

  afterEach(() => {
    if (dir) {
      rmSync(dir, { recursive: true, force: true })
      dir = undefined
    }
  })

  function app() {
    dir = mkdtempSync(join(tmpdir(), 'tiptap-dict-'))
    const path = join(dir, 'vocab.json')
    writeFileSync(
      path,
      vocabRow('important', '重要的；重大的', 'This is very important.', '这非常重要。') + '\n',
      'utf8',
    )
    const dictionaryService = createDictionaryService({ path })
    return createApp({
      aiService: buildAiService(),
      translateService: buildTranslateService(),
      dictionaryService,
    })
  }

  it('GET /api/dictionary/lookup 命中 → found:true', async () => {
    const res = await request(app()).get('/api/dictionary/lookup?word=important')
    expect(res.status).toBe(200)
    expect(res.body.found).toBe(true)
    expect(res.body.entry.word).toBe('important')
  })

  it('GET /api/dictionary/lookup 没命中 → 200 + found:false', async () => {
    const res = await request(app()).get('/api/dictionary/lookup?word=zzzz')
    expect(res.status).toBe(200)
    expect(res.body.found).toBe(false)
  })

  it('GET /api/dictionary/lookup 缺 word → 400', async () => {
    const res = await request(app()).get('/api/dictionary/lookup')
    expect(res.status).toBe(400)
  })

  it('GET /api/dictionary/status 返回路径和词数', async () => {
    const res = await request(app()).get('/api/dictionary/status')
    expect(res.status).toBe(200)
    expect(res.body.path).toContain('vocab.json')
    expect(typeof res.body.count).toBe('number')
  })
})
