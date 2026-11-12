import { describe, expect, it } from 'vitest'
import {
  buildContext,
  renderPrompt,
  type CapabilityInput,
} from '../src/config/capabilities.js'
import { defaultPromptSet } from '../src/config/defaults.js'
import type { CapabilityId } from '../src/config/types.js'

// 下面这些是**迁移前**的原始实现，从 git 里原样抽出来冻存在
// tests/legacy/prompts/ 下（见该目录的 README）。
//
// 之所以用真代码而不是把期望值抄进测试文件：抄的时候手一抖就把
// "do not include it in the output" 改成了 "... in your output"，
// 两边一起错，测试反而是绿的。冻代码才是真的对照。
import {
  buildGrammarCheckPrompt,
  GRAMMAR_SYSTEM,
} from './legacy/prompts/grammarCheck.js'
import {
  buildInstructionPrompt,
  INSTRUCTION_SYSTEM,
} from './legacy/prompts/rewriteInstruction.js'
import {
  buildRewriteSelectionPrompt,
  buildRewriteSelectionSystem,
  type SelectionKind,
} from './legacy/prompts/rewriteSelection.js'
import {
  buildRewriteSentencePrompt,
  buildRewriteSentenceSystem,
  type SentenceOperation,
} from './legacy/prompts/rewriteSentence.js'
import {
  buildTranslatePrompt,
  buildTranslateSystemPrompt,
} from './legacy/prompts/translate.js'
import type { ToneId } from '../src/config/defaults.js'

/**
 * 迁移安全网：默认提示词集渲染出来的结果，必须和迁移前的 builder
 * **逐字节相同**。
 */

const promptSet = defaultPromptSet()

function render(
  capability: CapabilityId,
  input: CapabilityInput,
): { system: string; user: string } {
  const template = promptSet.templates[capability]
  const context = buildContext(capability, input, promptSet.tones)
  return renderPrompt(template, context)
}

describe('提示词迁移的逐字节一致性', () => {
  describe('translate', () => {
    it('不带语气', () => {
      const out = render('translate', { text: '今天天气很好。' })
      expect(out.system).toBe(legacyTranslateSystem())
      expect(out.user).toBe(buildTranslatePrompt('今天天气很好。'))
    })

    it('带语气', () => {
      for (const tone of ['formal', 'casual', 'persuasive'] as ToneId[]) {
        const out = render('translate', { text: '今天天气很好。', tone })
        expect(out.system).toBe(legacyTranslateSystem(tone))
      }
    })
  })

  describe('rewriteSelection', () => {
    it('word，无上下文', () => {
      const out = render('rewriteSelection', { text: 'good', kind: 'word' })
      expect(out.system).toBe(buildRewriteSelectionSystem('word'))
      expect(out.user).toBe(buildRewriteSelectionPrompt('good'))
    })

    it('phrase，带上下文和中文原句', () => {
      const out = render('rewriteSelection', {
        text: 'very important',
        kind: 'phrase',
        context: 'Education is very important for children.',
        sourceSentence: '教育对儿童来说非常重要。',
      })
      expect(out.system).toBe(buildRewriteSelectionSystem('phrase'))
      expect(out.user).toBe(
        buildRewriteSelectionPrompt(
          'very important',
          'Education is very important for children.',
          '教育对儿童来说非常重要。',
        ),
      )
    })

    it('只有中文原句（无上下文）', () => {
      const out = render('rewriteSelection', {
        text: 'good',
        kind: 'word',
        sourceSentence: '很好。',
      })
      expect(out.user).toBe(
        buildRewriteSelectionPrompt('good', undefined, '很好。'),
      )
    })
  })

  describe('rewriteSentence', () => {
    it('rewrite 不带语气', () => {
      const out = render('rewriteSentence', {
        sentence: 'Education is very important for children.',
        operation: 'rewrite',
      })
      expect(out.system).toBe(buildRewriteSentenceSystem('rewrite'))
      expect(out.user).toBe(
        buildRewriteSentencePrompt('Education is very important for children.'),
      )
    })

    it('rewrite 带语气', () => {
      const out = render('rewriteSentence', {
        sentence: 'Education is very important for children.',
        operation: 'rewrite',
        tone: 'formal',
      })
      expect(out.system).toBe(buildRewriteSentenceSystem('rewrite', 'formal'))
    })

    it('其它操作即使传了语气也不带语气（和旧行为一致）', () => {
      for (const operation of ['improve', 'shorten', 'expand'] as const) {
        const out = render('rewriteSentence', {
          sentence: 'Education is very important for children.',
          operation,
          tone: 'formal',
        })
        expect(out.system).toBe(buildRewriteSentenceSystem(operation))
      }
    })

    it('带上下文和中文原句', () => {
      const out = render('rewriteSentence', {
        sentence: 'Education is very important for children.',
        operation: 'expand',
        context: 'Education shapes a child future.',
        sourceSentence: '教育对儿童来说非常重要。',
      })
      expect(out.user).toBe(
        buildRewriteSentencePrompt(
          'Education is very important for children.',
          'Education shapes a child future.',
          '教育对儿童来说非常重要。',
        ),
      )
    })
  })

  describe('grammarCheck', () => {
    it('原样透传句子', () => {
      const out = render('grammarCheck', {
        sentence: '  He go to school.  ',
      })
      expect(out.system).toBe(GRAMMAR_SYSTEM)
      expect(out.user).toBe(
        buildGrammarCheckPrompt('  He go to school.  '),
      )
    })
  })

  describe('rewriteInstruction', () => {
    it('不带上下文', () => {
      const out = render('rewriteInstruction', {
        text: 'Education is very important for children.',
        instruction: 'Make it more academic.',
      })
      expect(out.system).toBe(INSTRUCTION_SYSTEM)
      expect(out.user).toBe(
        buildInstructionPrompt(
          'Education is very important for children.',
          'Make it more academic.',
        ),
      )
    })

    it('带上下文', () => {
      const out = render('rewriteInstruction', {
        text: 'Education is very important for children.',
        instruction: 'Make it more academic.',
        context: 'Education shapes a child future.',
      })
      expect(out.user).toBe(
        buildInstructionPrompt(
          'Education is very important for children.',
          'Make it more academic.',
          'Education shapes a child future.',
        ),
      )
    })
  })

  describe('语气表', () => {
    it('默认语气表的 guidance 和迁移前一致', async () => {
      const legacy = await import('./legacy/prompts/tones.js')
      for (const id of legacy.TONE_IDS) {
        const tone = promptSet.tones.find((t) => t.id === id)
        expect(tone?.guidance).toBe(legacy.TONE_GUIDANCE[id])
      }
    })
  })
})

/** 旧 translate.ts 的 system builder（带 tone）—— 用旧 tones 表拼。 */
function legacyTranslateSystem(tone?: ToneId): string {
  return buildTranslateSystemPrompt(tone)
}

/** 类型断言：确认旧 builder 的签名没变（编译期检查）。 */
const _kindCheck: SelectionKind = 'word'
const _opCheck: SentenceOperation = 'rewrite'
void _kindCheck
void _opCheck
