import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { listWords } from '../lib/dictionary.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const out = path.join(__dirname, '..', 'public', 'words.json')

const words = listWords()
fs.writeFileSync(out, JSON.stringify(words))
console.log(`${words.length} words -> ${out}`)
