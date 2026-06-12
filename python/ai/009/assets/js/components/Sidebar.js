// Sidebar.js — 章节列表 + 进度标记
import { ref } from "vue";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const Sidebar = {
  setup() {
    const { state, actions } = useStore();
    const t = (key) => pick(I18N.ui[key] || { zh: key, en: key }, state.language);

    // 章节状态严格按当前语言查：ch[state.language].status
    function chapterStatus(id) {
      return state.progress?.chapters?.[id]?.[state.language]?.status || "locked";
    }
    function isCompleted(id) {
      return chapterStatus(id) === "completed";
    }
    function isLocked(id) {
      const meta = state.chaptersMap[id];
      if (!meta || !meta.prerequisites || meta.prerequisites.length === 0) return false;
      return meta.prerequisites.some((p) => !isCompleted(p));
    }
    function go(id) {
      if (isLocked(id)) return;
      window.location.hash = `#/ch/${id}`;
    }
    function isOptional(id) {
      return state.chaptersMap[id]?.is_optional || false;
    }
    function resetAll() {
      const msg = state.language === "en"
        ? "Reset all progress, quiz answers, and certificate? You will start over. (Theme and language preference are kept.)"
        : "重置所有进度、答题记录和证书？将重新开始学习。（保留主题和语言偏好）";
      if (!confirm(msg)) return;
      actions.resetAll();
    }
    // 调试：显示当前 localStorage 进度（让用户能确认中英是否分开）
    const debugOpen = ref(false);
    function showDebug() { debugOpen.value = true; }
    function closeDebug() { debugOpen.value = false; }
    function debugText() {
      const p = state.progress;
      if (!p) return "(no progress)";
      // 简化输出：每个章节的中英状态
      const lines = [];
      for (const meta of state.chapters) {
        const ch = p.chapters[meta.id];
        if (!ch) { lines.push(`${meta.id}: (空)`); continue; }
        const zh = ch.zh?.status || "—";
        const en = ch.en?.status || "—";
        const zhScroll = ch.zh?.scroll_percent ?? 0;
        const enScroll = ch.en?.scroll_percent ?? 0;
        const qZh = Object.keys(ch.quiz || {}).filter(k => k.endsWith(":zh")).length;
        const qEn = Object.keys(ch.quiz || {}).filter(k => k.endsWith(":en")).length;
        lines.push(`${meta.id}: zh=${zh} (scroll ${zhScroll}%, ${qZh}答) | en=${en} (scroll ${enScroll}%, ${qEn}答)`);
      }
      lines.push("");
      lines.push(`summary: ${JSON.stringify(p.summary)}`);
      return lines.join("\n");
    }

    return { state, t, chapterStatus, isCompleted, isLocked, go, isOptional, resetAll, debugOpen, showDebug, closeDebug, debugText };
  },
  template: `
    <aside class="sidebar">
      <h3>📚 {{ t('chaptersTitle') }}</h3>
      <a
        v-for="ch in state.chapters"
        :key="ch.id"
        :href="isLocked(ch.id) ? '#' : '#/ch/' + ch.id"
        :class="[
          'chapter-link',
          { active: $route.params.id === ch.id, completed: isCompleted(ch.id), locked: isLocked(ch.id) }
        ]"
        @click.prevent="go(ch.id)"
      >
        <span class="chapter-num">{{ String(ch.order).padStart(2, '0') }}</span>
        <span class="chapter-title">
          {{ state.language === 'en' ? ch.title_en : ch.title_zh }}
          <span v-if="isOptional(ch.id)" class="chapter-badge" style="background:var(--muted)">OPT</span>
          <span v-else-if="isCompleted(ch.id)" class="chapter-badge">✓</span>
        </span>
      </a>

      <h3 style="margin-top:24px">🏆 {{ t('certTitle') }}</h3>
      <a
        href="#/cert"
        class="chapter-link"
        :class="{ active: $route.name === 'cert' }"
      >
        <span class="chapter-num">🎓</span>
        <span class="chapter-title">{{ t('certLink') }}</span>
      </a>

      <div class="sidebar-footer">
        <button class="reset-btn" @click="showDebug">🔍 {{ state.language === 'en' ? 'Debug Progress' : '查看进度详情' }}</button>
        <button class="reset-btn" @click="resetAll" :title="t('resetTitle') || ''">
          🔄 {{ t('resetProgress') }}
        </button>
      </div>

      <div v-if="debugOpen" class="debug-modal" @click.self="closeDebug">
        <div class="debug-modal-content">
          <div class="debug-modal-header">
            <h3>{{ state.language === 'en' ? 'Progress Details' : '进度详情' }} (当前语言: {{ state.language }})</h3>
            <button class="debug-close" @click="closeDebug">×</button>
          </div>
          <pre class="debug-pre">{{ debugText() }}</pre>
        </div>
      </div>
    </aside>
  `,
};
