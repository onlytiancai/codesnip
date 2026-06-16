// HomeView.js — 欢迎页 + 章节卡片网格
import { useStore } from "../store.js";

export const HomeView = {
  setup() {
    const { state } = useStore();

    function isCompleted(id) {
      return state.progress?.chapters?.[id]?.status === "completed";
    }
    function isLocked(id) {
      const meta = state.chaptersMap[id];
      if (!meta?.prerequisites?.length) return false;
      return meta.prerequisites.some((p) => !isCompleted(p));
    }
    function go(id) {
      if (isLocked(id)) return;
      window.location.hash = `#/ch/${id}`;
    }

    const totalQuizzes = state.chapters.reduce((s, c) => s + (c.quiz_count || 0), 0);
    const totalMinutes = state.chapters.reduce((s, c) => s + (c.estimated_minutes || 0), 0);

    return { state, isCompleted, isLocked, go, totalQuizzes, totalMinutes };
  },
  template: `
    <div class="home-hero">
      <h1>{{ state.language === 'en' ? 'Neural Network 101' : '神经网络小课堂' }}</h1>
      <p class="tagline">
        {{ state.language === 'en'
          ? 'Hand-coding an MLP — the #1 AI interview question, explained for kids'
          : '面试 AI 岗第一题——手写 MLP，从零讲给小学生听' }}
      </p>
      <div class="stats">
        <div class="stat">
          <div class="val">{{ state.chapters.length }}</div>
          <div class="lbl">{{ state.language === 'en' ? 'Chapters' : '章节' }}</div>
        </div>
        <div class="stat">
          <div class="val">{{ totalQuizzes }}+</div>
          <div class="lbl">{{ state.language === 'en' ? 'Quizzes' : '测试题' }}</div>
        </div>
        <div class="stat">
          <div class="val">{{ totalMinutes }}</div>
          <div class="lbl">{{ state.language === 'en' ? 'Minutes' : '分钟' }}</div>
        </div>
      </div>
    </div>

    <h2 style="text-align:center; margin-top:24px">
      {{ state.language === 'en' ? 'Chapters' : '章节目录' }}
    </h2>
    <div class="chapter-grid">
      <a
        v-for="ch in state.chapters"
        :key="ch.id"
        :href="isLocked(ch.id) ? '#' : '#/ch/' + ch.id"
        :class="['chapter-card', { completed: isCompleted(ch.id) }]"
        @click.prevent="go(ch.id)"
      >
        <div class="num">
          CH {{ String(ch.order).padStart(2, '0') }}
          <span v-if="ch.is_optional" style="color:var(--muted)">· OPTIONAL</span>
        </div>
        <h3>{{ state.language === 'en' ? ch.title_en : ch.title_zh }}</h3>
        <p>{{ state.language === 'en' ? ch.summary_en : ch.summary_zh }}</p>
      </a>
    </div>
  `,
};
