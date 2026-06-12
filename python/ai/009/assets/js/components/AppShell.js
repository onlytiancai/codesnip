// AppShell.js — 顶层布局（header + sidebar + main 三段 grid）
import { Sidebar } from "./Sidebar.js";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const AppShell = {
  components: { Sidebar },
  setup() {
    const { state, actions } = useStore();
    const t = (key) => pick(I18N.ui[key] || { zh: key, en: key }, state.language);

    function changeLang(e) {
      actions.setLanguage(e.target.value);
    }
    function toggleTheme() {
      actions.toggleTheme();
    }

    return { state, t, changeLang, toggleTheme };
  },
  template: `
    <div class="app-shell">
      <header>
        <div class="brand">🧠 {{ state.language === 'en' ? 'Neural Network 101' : '神经网络小课堂' }}</div>

        <div class="progress-bar" v-if="state.progress">
          <div class="progress-stat">
            <div class="lbl">{{ t('chaptersTitle') }}</div>
            <div class="val">
              {{ Object.values(state.progress.chapters).filter(c => c.status === 'completed').length }}
              / {{ state.chapters.length }}
            </div>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{
              width: (Object.values(state.progress.chapters).filter(c => c.status === 'completed').length / Math.max(1, state.chapters.length) * 100) + '%'
            }"></div>
          </div>
          <div class="progress-stat">
            <div class="lbl">{{ t('correctRate') }}</div>
            <div class="val">{{ Math.round(quizPct() * 100) }}%</div>
          </div>
        </div>

        <div class="spacer"></div>

        <select :value="state.language" @change="changeLang" aria-label="Language">
          <option value="zh">中文</option>
          <option value="en">English</option>
        </select>

        <button class="ghost" @click="toggleTheme" :title="t('theme')" aria-label="Theme">
          {{ state.theme === 'light' ? '🌙' : '☀️' }}
        </button>
      </header>

      <div class="app-body">
        <Sidebar />
        <main class="content">
          <router-view :key="$route.fullPath" />
        </main>
      </div>
    </div>
  `,
  methods: {
    quizPct() {
      const s = this.state.progress?.summary;
      if (!s || !s.quizzes_total) return 0;
      return s.quizzes_correct / s.quizzes_total;
    },
  },
};
