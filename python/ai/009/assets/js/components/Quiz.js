// Quiz.js — 测试题组件
// props.data = parseQuizBlock() 的输出
// props.chapterId = 父章节 id
import { ref, computed, onMounted } from "vue";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const Quiz = {
  props: ["data", "chapterId"],
  setup(props) {
    const { state, actions } = useStore();
    const t = (k) => pick(I18N.ui[k] || { zh: k, en: k }, state.language);

    const selected = ref([]);     // 选中项 key 数组（多选复用）
    const submitted = ref(false);
    const showExplain = ref(false);
    const shortText = ref("");

    const isMultiple = props.data.type === "multiple";
    const isShort = props.data.type === "short";
    const isSingle = !isMultiple && !isShort;

    // 加载已答过的状态
    const existing = computed(() => {
      return state.progress?.chapters?.[props.chapterId]?.quiz?.[props.data.id] || null;
    });

    onMounted(() => {
      if (existing.value) {
        if (isShort) {
          shortText.value = existing.value.given || "";
          submitted.value = true;
          showExplain.value = true;
        } else {
          selected.value = (existing.value.given || "").split(",").filter(Boolean);
          submitted.value = true;
          showExplain.value = existing.value.correct === true;
        }
      }
    });

    const correctKeys = computed(() => {
      return (props.data.answer || "").split(",").map((s) => s.trim()).filter(Boolean);
    });

    const isOptionCorrect = (key) => correctKeys.value.includes(key);
    const isOptionWrong = (key) => submitted.value && selected.value.includes(key) && !isOptionCorrect(key);
    const isOptionMissed = (key) => submitted.value && isMultiple && isOptionCorrect(key) && !selected.value.includes(key);

    const isAnsweredCorrectly = computed(() => {
      if (!submitted.value) return false;
      if (isShort) return null;  // 简答不判分
      if (isSingle) return selected.value.length === 1 && isOptionCorrect(selected.value[0]);
      if (isMultiple) {
        const sel = new Set(selected.value);
        const cor = new Set(correctKeys.value);
        return sel.size === cor.size && [...sel].every((k) => cor.has(k));
      }
      return false;
    });

    function toggleOption(key) {
      if (submitted.value) return;
      if (isSingle) {
        selected.value = [key];
      } else if (isMultiple) {
        if (selected.value.includes(key)) {
          selected.value = selected.value.filter((k) => k !== key);
        } else {
          selected.value = [...selected.value, key];
        }
      }
    }

    function submit() {
      if (isShort) {
        if (!shortText.value.trim()) { showToast(t("emptyShort")); return; }
        submitted.value = true;
        showExplain.value = true;
        actions.answerQuiz(props.chapterId, props.data.id, {
          type: "short",
          given: shortText.value.trim(),
          correct: null,
          lang: state.language,
        });
        return;
      }
      if (selected.value.length === 0) { showToast(t("noAnswer")); return; }
      submitted.value = true;
      showExplain.value = true;
      const correct = isAnsweredCorrectly.value;
      actions.answerQuiz(props.chapterId, props.data.id, {
        type: props.data.type,
        given: selected.value.join(","),
        correct,
        lang: state.language,
      });
    }

    function reset() {
      selected.value = [];
      shortText.value = "";
      submitted.value = false;
      showExplain.value = false;
    }

    function showToast(msg) {
      const el = document.createElement("div");
      el.className = "toast";
      el.textContent = msg;
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 2000);
    }

    return {
      state, t,
      selected, submitted, showExplain, shortText,
      isMultiple, isShort, isSingle,
      existing, correctKeys,
      isOptionCorrect, isOptionWrong, isOptionMissed,
      isAnsweredCorrectly,
      toggleOption, submit, reset,
    };
  },
  template: `
    <div :class="['quiz', isMultiple ? 'q-multiple' : '', isShort ? 'q-short' : '']">
      <div v-if="existing && submitted" :class="['answered-badge', existing.correct === true ? 'correct' : (existing.correct === false ? 'wrong' : '')]">
        {{ t('answered') }}
      </div>

      <div class="quiz-prompt" v-html="renderInline(data.prompt)"></div>

      <!-- 单选 / 多选 -->
      <div v-if="!isShort" class="options">
        <div
          v-for="opt in data.options"
          :key="opt.key"
          :class="[
            'option',
            {
              selected: selected.includes(opt.key) && !submitted,
              correct: submitted && isOptionCorrect(opt.key),
              wrong: isOptionWrong(opt.key),
              disabled: submitted,
              'missed-correct': isOptionMissed(opt.key),
            }
          ]"
          :style="isOptionMissed(opt.key) ? 'border-style:dashed; border-color:var(--success)' : ''"
          @click="toggleOption(opt.key)"
        >
          <span class="key">{{ opt.key }}</span>
          <span class="text" v-html="renderInline(opt.text)"></span>
        </div>
      </div>

      <!-- 简答 -->
      <div v-else>
        <textarea
          v-model="shortText"
          class="short-answer"
          :name="'q-' + data.id"
          :id="'q-' + data.id"
          :aria-label="data.placeholder || (state.language === 'en' ? 'Short answer' : '简答')"
          :placeholder="data.placeholder || (state.language === 'en' ? 'Type your answer here...' : '在这里输入你的回答…')"
          :disabled="submitted"
        ></textarea>
      </div>

      <div class="actions">
        <button v-if="!submitted" class="primary" @click="submit">
          {{ t('submit') }}
        </button>
        <template v-else>
          <button @click="showExplain = !showExplain">
            {{ showExplain ? t('hideAnswer') : t('viewAnswer') }}
          </button>
          <button @click="reset">{{ t('reset') }}</button>
        </template>
      </div>

      <!-- 反馈 -->
      <div v-if="submitted && !isShort" :class="['feedback', isAnsweredCorrectly ? 'correct' : 'wrong']">
        <span v-if="isAnsweredCorrectly">✓ {{ state.language === 'en' ? 'Correct!' : '答对了！' }}</span>
        <span v-else>
          ✗ {{ state.language === 'en' ? 'Wrong' : '答错了' }}
          ·
          <span v-if="state.language === 'en'">Correct answer:</span>
          <span v-else>正确答案：</span>
          <strong>{{ correctKeys.join(', ') }}</strong>
        </span>
      </div>

      <!-- 答案解析（折叠） -->
      <div v-if="showExplain && data.explain" class="explain" v-html="renderMarkdownSafe(data.explain)"></div>
      <div v-if="showExplain && isShort && data.model_answer" class="explain">
        <div style="font-weight:700; color:var(--warn); margin-bottom:6px; font-size:12px; letter-spacing:0.5px">
          💡 {{ t('refAnswer') }}
        </div>
        <div v-html="renderMarkdownSafe(data.model_answer)"></div>
      </div>
    </div>
  `,
  methods: {
    // 简单 inline markdown：粗体、代码、$..$ KaTeX
    renderInline(text) {
      if (!text) return "";
      let s = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      // 行内公式
      if (typeof window.katex !== "undefined") {
        s = s.replace(/\$([^$]+)\$/g, (_, tex) => {
          try { return window.katex.renderToString(tex, { throwOnError: false, displayMode: false }); }
          catch (e) { return _; }
        });
      }
      return s;
    },
    renderMarkdownSafe(text) {
      if (!text) return "";
      let s = text
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/\n\n/g, "<br/><br/>")
        .replace(/\n/g, "<br/>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
      if (typeof window.katex !== "undefined") {
        s = s.replace(/\$([^$]+)\$/g, (_, tex) => {
          try { return window.katex.renderToString(tex, { throwOnError: false, displayMode: false }); }
          catch (e) { return _; }
        });
      }
      return s;
    },
  },
};
