import { createApp } from './app.js'
import { createOllamaService } from './services/ollama.js'

const PORT = Number(process.env.PORT ?? 8787)
const MODEL = process.env.OLLAMA_MODEL ?? 's2021008840/hy-mt2:1.8b'

const service = createOllamaService({ model: MODEL })
const app = createApp({ service })

app.listen(PORT, () => {
  // eslint-disable-next-line no-console
  console.log(`[tiptap-rewrite-server] listening on http://127.0.0.1:${PORT}`)
  // eslint-disable-next-line no-console
  console.log(`[tiptap-rewrite-server] model: ${MODEL}`)
})