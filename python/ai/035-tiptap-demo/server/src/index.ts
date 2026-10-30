import { createApp } from './app.js'
import { loadEnv } from './config/env.js'
import { ConfigStore } from './config/store.js'
import { ClientRegistry } from './llm/registry.js'
import { createActiveProviderGetter } from './services/activeProvider.js'
import { createAiService } from './services/ai.js'
import { createTranslateService } from './services/translate.js'
import { createDictionaryService } from './services/dictionary.js'

// 必须最先跑：PORT / OLLAMA_* / API key 都从 server/.env 里读，
// 而 defaultProvider()（在 new ConfigStore() 里）会读 OLLAMA_HOST。
loadEnv()

const PORT = Number(process.env.PORT ?? 8787)

const configStore = new ConfigStore()
const registry = new ClientRegistry()
const getActive = createActiveProviderGetter(configStore, registry)

const aiService = createAiService({ getActive })
const translateService = createTranslateService({ getActive })
const dictionaryService = createDictionaryService()
const app = createApp({
  aiService,
  translateService,
  configStore,
  registry,
  dictionaryService,
})

app.listen(PORT, () => {
  const provider = configStore.getActiveProvider()
  // eslint-disable-next-line no-console
  console.log(`[tiptap-rewrite-server] listening on http://127.0.0.1:${PORT}`)
  // eslint-disable-next-line no-console
  console.log(
    `[tiptap-rewrite-server] provider: ${provider.label} (${provider.kind} / ${provider.model})`,
  )
})
