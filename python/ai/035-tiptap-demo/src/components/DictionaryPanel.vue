<script setup lang="ts">
import { onMounted, ref } from 'vue'
import {
  fetchDictionaryStatus,
  reloadDictionary,
  type DictionaryStatus,
} from '../services/dictionary'
import { lookupWord, type LookupResult } from '../services/dictionary'

/**
 * 词库设置（只读）。
 *
 * 路径在 server/.env 的 VOCAB_PATH 里改 —— 这里只展示状态，
 * 加一个「重新加载」：后端会重读 .env 并重建索引，不用重启 server。
 */

const status = ref<DictionaryStatus | null>(null)
const loading = ref(false)
const error = ref('')

/** 测试查询 */
const probeWord = ref('important')
const probe = ref<LookupResult | null>(null)
const probing = ref(false)

async function load() {
  loading.value = true
  error.value = ''
  try {
    status.value = await fetchDictionaryStatus()
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function reload() {
  loading.value = true
  error.value = ''
  try {
    status.value = await reloadDictionary()
    probe.value = null
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    loading.value = false
  }
}

async function runProbe() {
  const word = probeWord.value.trim()
  if (!word) return
  probing.value = true
  try {
    probe.value = await lookupWord(word)
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  } finally {
    probing.value = false
  }
}

onMounted(load)
</script>

<template>
  <div class="dictionary">
    <p class="dictionary__hint">
      查词用的是本地词库，不经过模型 —— 选中英文里的一个单词就能查释义、例句。
      词库路径在 <code>server/.env</code> 的 <code>VOCAB_PATH</code> 里改，
      改完点「重新加载」即可（不用重启后端）。
    </p>

    <div class="dictionary__row">
      <label class="field field--inline">
        <span class="field__label">词库路径</span>
        <input class="field__input" :value="status?.path ?? ''" readonly />
      </label>
      <button class="btn" :disabled="loading" @click="reload">重新加载</button>
    </div>

    <p v-if="status" class="dictionary__status">
      <template v-if="status.loaded">
        已加载 <strong>{{ status.count }}</strong> 个词
      </template>
      <template v-else> 词库未加载：{{ status.error ?? '未知原因' }} </template>
    </p>

    <p v-if="error" class="settings__error">{{ error }}</p>

    <div class="dictionary__probe">
      <label class="field field--inline">
        <span class="field__label">测试查询</span>
        <input
          v-model="probeWord"
          class="field__input"
          placeholder="important"
          @keyup.enter="runProbe"
        />
      </label>
      <button class="btn" :disabled="probing" @click="runProbe">查询</button>
    </div>

    <div v-if="probe" class="dictionary__result">
      <template v-if="probe.found && probe.entry">
        <p class="dictionary__word">
          <strong>{{ probe.entry.word }}</strong>
          <span v-if="probe.entry.ukphone || probe.entry.usphone" class="dictionary__phonetic">
            /{{ probe.entry.ukphone || probe.entry.usphone }}/
          </span>
        </p>
        <ul v-if="probe.entry.trans.length" class="dictionary__list">
          <li v-for="(t, i) in probe.entry.trans" :key="i">
            <span v-if="t.pos" class="dictionary__pos">{{ t.pos }}</span>{{ t.cn }}
          </li>
        </ul>
        <ul v-if="probe.entry.sentences.length" class="dictionary__list">
          <li v-for="(s, i) in probe.entry.sentences" :key="`s${i}`">
            {{ s.en }}<span v-if="s.cn" class="dictionary__cn"> {{ s.cn }}</span>
          </li>
        </ul>
      </template>
      <p v-else class="dictionary__missing">
        {{ probe.error ?? `词典里没有「${probe.word}」` }}
      </p>
    </div>
  </div>
</template>
