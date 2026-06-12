// AppShell.js — 顶层布局（header + sidebar + main 三段 grid）
import { computed } from "vue";
import { Sidebar } from "./Sidebar.js";
import { useStore } from "../store.js";
import { I18N, pick } from "../i18n.js";

export const AppShell = {
  components: { Sidebar },
  setup() {
    const { state, actions } = useStore();
    const t = (key) => pick(I18N.ui[key] || { zh: key, en: key }, state.language);

    // 显式 computed：当 state.language 或 state.progress.chapters 变化时重算
    // 严格按当前语言 ch[state.language].status === 'completed' 过滤
    const completedCount = computed(() => {
      const chapters = state.progress?.chapters || {};
      let n = 0;
      for (const ch of Object.values(chapters)) {
        if (ch[state.language]?.status === "completed") n++;
      }
      return n;
    });
    const progressPct = computed(() => {
      const total = Math.max(1, state.chapters.length);
      return (completedCount.value / total) * 100;
    });
    const quizPctValue = computed(() => {
      const s = state.progress?.summary;
      if (!s || !s.quizzes_total) return 0;
      return s.quizzes_correct / s.quizzes_total;
    });

    function changeLang(e) {
      actions.setLanguage(e.target.value);
    }
    function toggleTheme() {
      actions.toggleTheme();
    }

    return { state, t, changeLang, toggleTheme, completedCount, progressPct, quizPctValue };
  },
  template: `
    <div class="app-shell">
      <header>
        <div class="brand">🧠 {{ state.language === 'en' ? 'Neural Network 101' : '神经网络小课堂' }} <span style="font-size:10px; opacity:0.5; margin-left:4px">v3</span></div>

        <div class="progress-bar" v-if="state.progress">
          <div class="progress-stat">
            <div class="lbl">{{ t('chaptersTitle') }}</div>
            <div class="val">
              {{ completedCount }} / {{ state.chapters.length }}
            </div>
          </div>
          <div class="progress-track">
            <div class="progress-fill" :style="{
              width: progressPct + '%'
            }"></div>
          </div>
          <div class="progress-stat">
            <div class="lbl">{{ t('correctRate') }}</div>
            <div class="val">{{ Math.round(quizPctValue * 100) }}%</div>
          </div>
        </div>

        <div class="spacer"></div>

        <select :value="state.language" @change="changeLang" name="language" id="language" aria-label="Language">
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
};
