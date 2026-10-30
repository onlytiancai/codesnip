/**
 * 查词：调后端的本地词库。
 *
 * 词库在后端（server/.env 的 VOCAB_PATH 指的那个 JSONL 文件），
 * 所以这边一次请求就拿到词条 —— 不经过模型，零 API 开销。
 */

import { http } from './http'

export interface DictionaryTrans {
  pos: string
  cn: string
  en: string
}

export interface DictionaryEntry {
  word: string
  usphone?: string
  ukphone?: string
  trans: DictionaryTrans[]
  sentences: { en: string; cn: string }[]
  phrases: { en: string; cn: string }[]
  synonyms: { pos: string; tran: string; words: string[] }[]
  antonyms: string[]
  mnemonic?: string
}

export interface DictionaryStatus {
  path: string
  count: number
  loaded: boolean
  error?: string
}

export interface LookupResult {
  found: boolean
  word: string
  entry?: DictionaryEntry
  /** 词库没加载起来时的原因 */
  error?: string
}

/** 查一个单词。没查到不算错 —— found:false。 */
export async function lookupWord(
  word: string,
  signal?: AbortSignal,
): Promise<LookupResult> {
  const body = await http.get<LookupResult>(
    `/api/dictionary/lookup?word=${encodeURIComponent(word)}`,
    signal,
  )
  return {
    found: body.found === true,
    word: body.word ?? word,
    entry: body.entry,
    error: body.error,
  }
}

/** 当前词库状态（路径 + 词数 + 有没有加载成功）。 */
export const fetchDictionaryStatus = (signal?: AbortSignal) =>
  http.get<DictionaryStatus>('/api/dictionary/status', signal)

/**
 * 重新读 server/.env + 重建索引。
 * 改了 .env 里的 VOCAB_PATH 之后，不用重启 server，点一下就行。
 */
export const reloadDictionary = (signal?: AbortSignal) =>
  http.post<DictionaryStatus>('/api/dictionary/reload', undefined, signal)
