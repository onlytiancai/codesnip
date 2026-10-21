#!/usr/bin/env node
/**
 * 手工测试 CLI：绕过 HTTP 路由，直接调 createOllamaService / createTranslateService。
 *
 * 用途：
 * - 单独验证某个 prompt 在本地模型上的表现（不用起 server）
 * - 看清楚「实际发给模型的 system / user 消息」与「模型原始输出」
 * - 复现解析失败（parseCandidates / cleanText）时看到底是哪一步炸的
 *
 * 用法见 `pnpm cli --help`。
 */

import { readFileSync } from 'node:fs'
import { createInterface } from 'node:readline/promises'
import { stdin, stdout } from 'node:process'
import ollamaDefault, { Ollama, type ChatRequest } from 'ollama'

import { createOllamaService } from './services/ollama.js'
import { createTranslateService } from './services/translate.js'
import {
  isSelectionKind,
  SELECTION_KINDS,
  type SelectionKind,
} from './prompts/rewriteSelection.js'
import {
  isSentenceOperation,
  SENTENCE_OPERATIONS,
  type SentenceOperation,
} from './prompts/rewriteSentence.js'
import { isToneId, TONE_IDS, type ToneId } from './prompts/tones.js'

// ---------------------------------------------------------------- 参数解析

type Flags = Record<string, string | boolean>

function parseArgs(argv: string[]): { cmd: string; flags: Flags; positional: string[] } {
  const flags: Flags = {}
  const positional: string[] = []
  let i = 0
  while (i < argv.length) {
    const arg = argv[i]
    if (arg.startsWith('--')) {
      const body = arg.slice(2)
      const eq = body.indexOf('=')
      if (eq >= 0) {
        flags[body.slice(0, eq)] = body.slice(eq + 1)
        i += 1
      } else if (i + 1 < argv.length && !argv[i + 1].startsWith('--')) {
        flags[body] = argv[i + 1]
        i += 2
      } else {
        flags[body] = true
        i += 1
      }
    } else {
      positional.push(arg)
      i += 1
    }
  }
  const cmd = positional.shift() ?? ''
  return { cmd, flags, positional }
}

function str(flags: Flags, key: string): string | undefined {
  const v = flags[key]
  if (typeof v === 'string') return v
  if (v === true) return ''
  return undefined
}

function num(flags: Flags, key: string): number | undefined {
  const v = str(flags, key)
  if (v === undefined || v === '') return undefined
  const n = Number(v)
  return Number.isFinite(n) ? n : undefined
}

function bool(flags: Flags, key: string): boolean {
  const v = flags[key]
  if (v === undefined) return false
  if (typeof v === 'boolean') return v
  return v !== 'false' && v !== '0'
}

// ---------------------------------------------------------------- 输入读取

/** --text / --file / 位置参数 / stdin / 交互输入，按优先级取。 */
async function readText(
  flags: Flags,
  positional: string[],
  label: string,
): Promise<string> {
  const fromFlag = str(flags, 'text')
  if (fromFlag) return fromFlag

  const fromFile = str(flags, 'file')
  if (fromFile) return readFileSync(fromFile, 'utf8')

  if (positional.length > 0) return positional.join(' ')

  if (!stdin.isTTY) {
    const chunks: Buffer[] = []
    for await (const chunk of stdin) chunks.push(Buffer.from(chunk))
    const piped = Buffer.concat(chunks).toString('utf8').trim()
    if (piped) return piped
    // 非 TTY 且没内容：没有交互机会，直接报错而不是干等
    throw new Error(`没有输入：请用 --text / --file 传入${label}`)
  }

  const rl = createInterface({ input: stdin, output: stdout })
  try {
    const answer = await rl.question(`${label}: `)
    return answer.trim()
  } finally {
    rl.close()
  }
}

async function readLine(label: string): Promise<string> {
  if (!stdin.isTTY) throw new Error(`没有输入：请用对应参数传入${label}`)
  const rl = createInterface({ input: stdin, output: stdout })
  try {
    const answer = await rl.question(`${label}: `)
    return answer.trim()
  } finally {
    rl.close()
  }
}

// ---------------------------------------------------------------- 录音 client

interface Recording {
  client: Ollama
  /** 真实 client（--host 生效的那一个），models 命令用 */
  base: Ollama
  /** 实际发给模型的请求（含 system / user 两条 message） */
  sent: ChatRequest[]
  /** 模型原始输出（未经 parseCandidates / cleanText） */
  raw: string[]
}

/**
 * 包一层 client，把「发了什么」「回了什么」都录下来。
 * createOllamaService 只用到 .chat，所以这里只实现这一个方法。
 */
function createRecordingClient(host?: string): Recording {
  const base = host ? new Ollama({ host }) : ollamaDefault
  const sent: ChatRequest[] = []
  const raw: string[] = []

  const client = {
    async chat(request: ChatRequest) {
      sent.push(request)
      const res = await base.chat({ ...request, stream: false })
      raw.push(res.message?.content ?? '')
      return res
    },
  } as unknown as Ollama

  return { client, base, sent, raw }
}

function splitMessages(request: ChatRequest | undefined): { system: string; user: string } {
  const messages = request?.messages ?? []
  const system = messages
    .filter((m) => m.role === 'system')
    .map((m) => m.content)
    .join('\n')
  const user = messages
    .filter((m) => m.role === 'user')
    .map((m) => m.content)
    .join('\n')
  return { system, user }
}

// ---------------------------------------------------------------- 输出

const DIM = '\u001b[2m'
const CYAN = '\u001b[36m'
const GREEN = '\u001b[32m'
const RED = '\u001b[31m'
const RESET = '\u001b[0m'

function heading(title: string): void {
  console.log(`\n${CYAN}── ${title} ──${RESET}`)
}

function printBlock(title: string, body: string): void {
  heading(title)
  console.log(body)
}

function printResult(value: unknown): void {
  if (Array.isArray(value)) {
    value.forEach((item, i) => console.log(`  ${GREEN}${i + 1}.${RESET} ${item}`))
  } else {
    console.log(`  ${GREEN}${String(value)}${RESET}`)
  }
}

// ---------------------------------------------------------------- 各命令

const USAGE = `
手工测试 CLI（直接调 service，不经过 HTTP 路由）

用法:
  pnpm cli <command> [options]

命令:
  selection     选区候选词（AiService.rewriteSelection）
  sentence      整句改写（AiService.rewriteSentence）
  grammar       语法检查（AiService.grammarCheck）
  instruction   自由指令改写（AiService.rewriteInstruction）
  translate     中译英（TranslateService.translate）
  raw           直接发 system/user 给模型，不做任何解析
  models        列出本地 Ollama 模型

通用选项:
  --model <name>    模型名（默认取 OLLAMA_MODEL，再默认 s2021008840/hy-mt2:1.8b）
  --host <url>      Ollama 地址（默认 http://127.0.0.1:11434）
  --text <str>      输入文本；不给则读位置参数 → stdin → 交互输入
  --file <path>     从文件读输入（覆盖 --text）
  --prompt          打印实际发出的 system / user 消息
  --raw             打印模型原始输出（未经解析）
  --json            以 JSON 输出全部结果
  --quiet           不打印装饰性内容（与 --json 一起用）

selection:
  --kind word|phrase        选区类型，默认 word
  --context <str>           上下文段落
  --source <str>            中文原句
  --max <n>                 候选上限，默认 5

sentence:
  --operation rewrite|improve|shorten|expand   默认 rewrite
  --tone <id>               语气（只有 rewrite 生效）
  --context <str>           上下文段落
  --source <str>            中文原句

grammar:
  （无额外选项）

instruction:
  --instruction <str>       指令，必填

translate:
  --tone <id>               语气

raw:
  --system <str>            system 消息
  --user <str>              user 消息，必填

示例:
  pnpm cli selection --text "very important" --kind phrase --context "Education is very important for children."
  pnpm cli sentence --text "Education is very important for children." --operation rewrite --tone formal
  pnpm cli grammar --text "He go to school last yesterday."
  echo "教育对孩子很重要。" | pnpm cli translate
  pnpm cli models
`

async function run(): Promise<void> {
  const { cmd, flags, positional } = parseArgs(process.argv.slice(2))

  if (!cmd || bool(flags, 'help') || cmd === 'help') {
    console.log(USAGE)
    return
  }

  const model =
    str(flags, 'model') || process.env.OLLAMA_MODEL || 's2021008840/hy-mt2:1.8b'
  const host = str(flags, 'host') || process.env.OLLAMA_HOST || 'http://127.0.0.1:11434'
  const quiet = bool(flags, 'quiet')
  const asJson = bool(flags, 'json')

  const rec = createRecordingClient(host)
  const ai = createOllamaService({
    client: rec.client,
    model,
    defaultMaxCandidates: num(flags, 'max') ?? 5,
  })
  const translator = createTranslateService({ client: rec.client, model })

  const started = Date.now()
  let result: unknown
  let error: string | undefined

  try {
    switch (cmd) {
      case 'selection': {
        const text = await readText(flags, positional, '选区文本')
        const kindFlag = str(flags, 'kind') ?? 'word'
        if (!isSelectionKind(kindFlag)) {
          throw new Error(`--kind 必须是 ${SELECTION_KINDS.join(' | ')}，收到 "${kindFlag}"`)
        }
        const kind: SelectionKind = kindFlag
        result = await ai.rewriteSelection(text, kind, {
          context: str(flags, 'context'),
          sourceSentence: str(flags, 'source'),
          maxCandidates: num(flags, 'max'),
        })
        break
      }
      case 'sentence': {
        const sentence = await readText(flags, positional, '句子')
        const opFlag = str(flags, 'operation') ?? 'rewrite'
        if (!isSentenceOperation(opFlag)) {
          throw new Error(
            `--operation 必须是 ${SENTENCE_OPERATIONS.join(' | ')}，收到 "${opFlag}"`,
          )
        }
        const operation: SentenceOperation = opFlag
        const tone = str(flags, 'tone')
        if (tone !== undefined && !isToneId(tone)) {
          throw new Error(`--tone 必须是 ${TONE_IDS.join(' | ')}，收到 "${tone}"`)
        }
        result = await ai.rewriteSentence(sentence, {
          operation,
          tone: tone as ToneId | undefined,
          context: str(flags, 'context'),
          sourceSentence: str(flags, 'source'),
        })
        break
      }
      case 'grammar': {
        const sentence = await readText(flags, positional, '句子')
        result = await ai.grammarCheck(sentence)
        break
      }
      case 'instruction': {
        const text = await readText(flags, positional, '待改写文本')
        const instruction = str(flags, 'instruction') || (await readLine('改写指令'))
        if (!instruction) throw new Error('缺少 --instruction')
        result = await ai.rewriteInstruction(text, { instruction, context: str(flags, 'context') })
        break
      }
      case 'translate': {
        const text = await readText(flags, positional, '中文')
        const tone = str(flags, 'tone')
        if (tone !== undefined && !isToneId(tone)) {
          throw new Error(`--tone 必须是 ${TONE_IDS.join(' | ')}，收到 "${tone}"`)
        }
        const out = await translator.translate(text, { tone: tone as ToneId | undefined })
        result = out.translation
        break
      }
      case 'raw': {
        const user = str(flags, 'user') || (await readLine('user 消息'))
        if (!user) throw new Error('缺少 --user')
        const system = str(flags, 'system') ?? ''
        const res = await rec.client.chat({
          model,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: user },
          ],
        })
        result = res.message?.content ?? ''
        break
      }
      case 'models': {
        const list = await rec.base.list()
        result = list.models.map((m) => m.name)
        break
      }
      default:
        throw new Error(`未知命令 "${cmd}"，用 --help 看用法`)
    }
  } catch (err) {
    error = err instanceof Error ? err.message : String(err)
  }

  const elapsed = Date.now() - started
  const { system, user } = splitMessages(rec.sent[0])
  const raw = rec.raw[0]

  if (asJson) {
    console.log(
      JSON.stringify(
        {
          command: cmd,
          model,
          host,
          prompt: rec.sent[0] ? { system, user } : undefined,
          raw,
          result,
          error,
          ms: elapsed,
        },
        null,
        2,
      ),
    )
    if (error) process.exitCode = 1
    return
  }

  if (!quiet) {
    console.log(`${DIM}command: ${cmd}${RESET}`)
    console.log(`${DIM}model:   ${model}${RESET}`)
    console.log(`${DIM}host:    ${host}${RESET}`)
  }

  if (bool(flags, 'prompt') && rec.sent[0]) {
    printBlock('system', system)
    printBlock('user', user)
  }

  if (bool(flags, 'raw') && raw !== undefined) {
    printBlock('raw (模型原始输出)', raw)
  }

  if (error) {
    heading('result')
    console.log(`${RED}✗ ${error}${RESET}`)
    process.exitCode = 1
    return
  }

  heading('result')
  printResult(result)
  console.log(`\n${DIM}${elapsed} ms${RESET}`)
}

run().catch((err) => {
  console.error(`${RED}✗ ${err instanceof Error ? err.message : String(err)}${RESET}`)
  process.exitCode = 1
})
