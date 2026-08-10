import express from 'express'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { listWords, detailOf } from './lib/dictionary.js'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

const app = express()
const PORT = process.env.PORT || 3000

app.use(express.static(path.join(__dirname, 'public')))

app.get('/api/words', (req, res) => {
  res.json(listWords())
})

app.get('/api/words/:head', (req, res) => {
  const d = detailOf(req.params.head)
  if (!d) return res.status(404).json({ error: 'not found' })
  res.json(d)
})

app.listen(PORT, () => {
  console.log(`http://localhost:${PORT}`)
})
