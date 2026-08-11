import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
// CET4_2.json is JSON Lines: one JSON object per line
const RAW = fs.readFileSync(path.join(__dirname, '..', 'CET4_2.json'), 'utf8')

const BY_HEAD = new Map()
for (const line of RAW.split('\n')) {
  if (!line.trim()) continue
  const c = JSON.parse(line)?.content?.word
  if (!c?.wordHead) continue
  BY_HEAD.set(c.wordHead, c)
}

function mapTrans(trans) {
  return (trans || []).map(t => ({
    pos: t.pos || '',
    cn: t.tranCn || '',
    other: t.tranOther || '',
  }))
}

function listWords() {
  return [...BY_HEAD.values()].map(c => ({
    word: c.wordHead,
    usphone: c.content?.usphone || '',
    ukphone: c.content?.ukphone || '',
    phone: c.content?.phone || '',
    trans: (c.content?.trans || []).map(t => ({
      pos: t.pos || '',
      cn: t.tranCn || '',
    })),
  }))
}

function detailOf(head) {
  const c = BY_HEAD.get(head)
  if (!c) return null
  const x = c.content || {}
  return {
    word: c.wordHead,
    usphone: x.usphone || '',
    ukphone: x.ukphone || '',
    phone: x.phone || '',
    trans: mapTrans(x.trans),
    sentence: (x.sentence?.sentences || []).map(s => ({
      en: s.sContent || '',
      cn: s.sCn || '',
    })),
    phrase: (x.phrase?.phrases || []).map(p => ({
      en: p.pContent || '',
      cn: p.pCn || '',
    })),
    syno: (x.syno?.synos || []).map(s => ({
      pos: s.pos || '',
      tran: s.tran || '',
      hwds: (s.hwds || []).map(h => h.w).filter(Boolean),
    })),
    remMethod: x.remMethod?.val || '',
  }
}

export { listWords, detailOf }
