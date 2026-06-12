// Sidebar.js — 章节列表 + 进度标记
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const Sidebar = {
  setup() {
    const { state } = useStore();

    function chapterStatus(id) {
      return state.progress?.chapters?.[id]?.status || "locked";
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

    return { state, chapterStatus, isCompleted, isLocked, go, isOptional };
  },
  template: `
    <aside class="sidebar">
      <h3>📚 章节目录</h3>
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

      <h3 style="margin-top:24px">🏆 结业</h3>
      <a
        href="#/cert"
        class="chapter-link"
        :class="{ active: $route.name === 'cert' }"
      >
        <span class="chapter-num">🎓</span>
        <span class="chapter-title">结业证书 / Certificate</span>
      </a>
    </aside>
  `,
};
