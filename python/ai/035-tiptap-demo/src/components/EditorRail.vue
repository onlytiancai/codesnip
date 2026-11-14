<script setup lang="ts">
/**
 * 右侧工具栏（rail）。
 *
 * 装着**页面级**的控制项：翻译语气、实时翻译开关，以及几个行为开关
 * （高亮哪几层、双击查词、单击候选词）。
 *
 * 可收缩：收缩后只剩一列图标（hover 出 tooltip），展开后带文字说明。
 * 收缩状态存 localStorage —— 刷新后还是收着的。
 */
import { computed, ref, watch } from 'vue'
import AppIcon from './AppIcon.vue'
import type { AppIconName } from './appIcons'
import { isToneId, toneLabel, type ToneOption } from '../constants/tones'

const props = defineProps<{
  /** 语气列表（label 从后端拉） */
  tones: ToneOption[]
  /** 当前方向不支持语气（英译中就是普通翻译）→ 语气下拉禁用 */
  toneDisabled?: boolean
}>()

/** 翻译语气；空串 = 不指定，走后端默认 prompt */
const tone = defineModel<string>('tone', { default: '' })
/** 实时翻译开关 */
const liveTranslate = defineModel<boolean>('liveTranslate', { default: false })
/** 高亮当前单词 */
const highlightWord = defineModel<boolean>('highlightWord', { default: true })
/** 高亮当前句子 */
const highlightSentence = defineModel<boolean>('highlightSentence', { default: true })
/** 高亮当前段落 */
const highlightParagraph = defineModel<boolean>('highlightParagraph', { default: true })
/** 双击单词 → 直接查词 */
const doubleClickLookup = defineModel<boolean>('doubleClickLookup', { default: true })
/** 单击单词 → 弹出其它候选词 */
const clickCandidates = defineModel<boolean>('clickCandidates', { default: true })

const COLLAPSE_KEY = 'tiptap-demo.railCollapsed'

function readCollapsed(): boolean {
  try {
    return localStorage.getItem(COLLAPSE_KEY) === '1'
  } catch {
    return false
  }
}

const collapsed = ref(readCollapsed())

watch(collapsed, (v) => {
  try {
    localStorage.setItem(COLLAPSE_KEY, v ? '1' : '0')
  } catch {
    // 存不了就算了
  }
})

/** 当前语气的中文名（收缩态的 tooltip 用） */
const toneTip = computed(() => {
  if (props.toneDisabled) return '英译中不支持语气'
  const t = tone.value
  return t && isToneId(t) ? `语气：${toneLabel(t)}` : '语气'
})

/** 三层高亮的开关 —— 「按下去 = 开」 */
const highlightToggles = [
  { icon: 'type' as AppIconName, label: '单词', tip: '高亮当前单词', model: highlightWord },
  { icon: 'quote' as AppIconName, label: '句子', tip: '高亮当前句子', model: highlightSentence },
  {
    icon: 'text-align-start' as AppIconName,
    label: '段落',
    tip: '高亮当前段落',
    model: highlightParagraph,
  },
]

/** 查词相关的开关 */
const lookupToggles = [
  {
    icon: 'book-open' as AppIconName,
    label: '双击查词',
    tip: '双击单词直接查词',
    model: doubleClickLookup,
  },
  {
    icon: 'replace' as AppIconName,
    label: '单击候选',
    tip: '单击单词弹出其它候选词',
    model: clickCandidates,
  },
]
</script>

<template>
  <aside class="rail" :class="{ 'rail--collapsed': collapsed }">
    <div class="rail__head">
      <span v-if="!collapsed" class="rail__title">工具栏</span>
      <button
        type="button"
        class="rail__toggle"
        :data-tip="collapsed ? '展开工具栏' : '收起工具栏'"
        :aria-label="collapsed ? '展开工具栏' : '收起工具栏'"
        :aria-expanded="!collapsed"
        @click="collapsed = !collapsed"
      >
        <AppIcon :name="collapsed ? 'panel-right-open' : 'panel-right-close'" :size="16" />
      </button>
    </div>

    <div class="rail__body">
      <!-- 语气 -->
      <div class="rail__row rail__row--tone">
        <template v-if="collapsed">
          <button
            type="button"
            class="rail__btn"
            :data-tip="toneTip"
            aria-label="展开工具栏选择语气"
            @click="collapsed = false"
          >
            <AppIcon name="sliders-horizontal" :size="16" />
          </button>
        </template>
        <template v-else>
          <span class="rail__label">语气</span>
          <select v-model="tone" class="rail__select" :disabled="props.toneDisabled">
            <option value="">默认语气</option>
            <option v-for="t in props.tones" :key="t.id" :value="t.id">{{ t.label }}</option>
          </select>
        </template>
      </div>

      <!-- 实时翻译 -->
      <button
        type="button"
        class="rail__btn"
        :class="{ 'is-on': liveTranslate }"
        :aria-pressed="liveTranslate"
        :data-tip="collapsed ? (liveTranslate ? '实时翻译已开启' : '实时翻译已关闭') : undefined"
        @click="liveTranslate = !liveTranslate"
      >
        <AppIcon name="zap" :size="16" />
        <span v-if="!collapsed" class="rail__text">实时翻译</span>
      </button>

      <div class="rail__sep" />

      <span v-if="!collapsed" class="rail__label rail__label--group">高亮</span>

      <button
        v-for="t in highlightToggles"
        :key="t.label"
        type="button"
        class="rail__btn"
        :class="{ 'is-on': t.model.value }"
        :aria-pressed="t.model.value"
        :data-tip="collapsed ? t.tip : undefined"
        @click="t.model.value = !t.model.value"
      >
        <AppIcon :name="t.icon" :size="16" />
        <span v-if="!collapsed" class="rail__text">{{ t.label }}</span>
      </button>

      <div class="rail__sep" />

      <span v-if="!collapsed" class="rail__label rail__label--group">查词</span>

      <button
        v-for="t in lookupToggles"
        :key="t.label"
        type="button"
        class="rail__btn"
        :class="{ 'is-on': t.model.value }"
        :aria-pressed="t.model.value"
        :data-tip="collapsed ? t.tip : undefined"
        @click="t.model.value = !t.model.value"
      >
        <AppIcon :name="t.icon" :size="16" />
        <span v-if="!collapsed" class="rail__text">{{ t.label }}</span>
      </button>
    </div>
  </aside>
</template>
