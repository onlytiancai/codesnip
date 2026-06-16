/* ============================================
   app.js — Vue 3 应用
   铁鹰式期权学习测试
   ============================================ */
(function () {
  'use strict';

  const { createApp, reactive, computed, ref, onMounted, watch, nextTick } = Vue;

  // marked 配置
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
  }

  // ============================================
  // 状态（全局）
  // ============================================
  const state = reactive({
    loading: true,
    parseError: null,
    questions: [],
    sections: [],
    answers: {},          // { [qid]: string[] }
    submitted: false,
    results: null,        // { perQuestion, perSection, overall }
    currentSectionIdx: 0,
  });

  // ============================================
  // 工具函数
  // ============================================
  function setEq(a, b) {
    if (a.length !== b.length) return false;
    const sa = [...a].sort();
    const sb = [...b].sort();
    return sa.every((v, i) => v === sb[i]);
  }

  function pickAnswer(qid, key) {
    if (state.submitted) return;
    const q = state.questions.find(x => x.id === qid);
    if (!q) return;
    const cur = state.answers[qid] || [];
    if (q.type === 'single') {
      state.answers[qid] = [key];
    } else {
      state.answers[qid] = cur.includes(key)
        ? cur.filter(k => k !== key)
        : [...cur, key];
    }
  }

  function submit() {
    const answeredCount = Object.values(state.answers).filter(a => a && a.length > 0).length;
    console.log('[Quiz] submit() called, answered =', answeredCount, '/', state.questions.length);
    if (answeredCount === 0) {
      console.warn('[Quiz] submit() aborted: no answers yet');
      return;
    }

    const perQuestion = {};
    const perSection = [];
    let totalCorrect = 0;
    let totalPoints = 0;
    let totalPointsGot = 0;
    let totalAnswered = 0;

    for (const s of state.sections) {
      let sCorrect = 0, sPoints = 0, sPointsGot = 0, sAnswered = 0, sPartial = 0;
      for (const qid of s.questionIds) {
        const q = state.questions.find(x => x.id === qid);
        if (!q) continue;
        const picked = state.answers[qid] || [];
        const right = q.answer;
        const correct = picked.length > 0 && setEq(picked, right);
        const partial = !correct && picked.length > 0 && picked.some(k => right.includes(k));
        perQuestion[qid] = { correct, picked, right, partial };
        totalPoints += q.points;
        sPoints += q.points;
        if (picked.length > 0) totalAnswered++, sAnswered++;
        if (correct) {
          totalCorrect++;
          totalPointsGot += q.points;
          sCorrect++;
          sPointsGot += q.points;
        } else if (partial) {
          const partialPts = Math.round(q.points * 0.5 * 10) / 10;
          totalPointsGot += partialPts;
          sPointsGot += partialPts;
          sPartial++;
        }
      }
      perSection.push({
        key: s.key,
        title: s.title,
        total: s.questionIds.length,
        correct: sCorrect,
        partial: sPartial,
        answered: sAnswered,
        rate: s.questionIds.length ? sCorrect / s.questionIds.length : 0,
        points: sPoints,
        pointsGot: sPointsGot,
      });
    }

    state.results = {
      perQuestion,
      perSection,
      overall: {
        total: state.questions.length,
        correct: totalCorrect,
        answered: totalAnswered,
        rate: state.questions.length ? totalCorrect / state.questions.length : 0,
        points: totalPoints,
        pointsGot: totalPointsGot,
        pointsRate: totalPoints ? totalPointsGot / totalPoints : 0,
      },
    };
    state.submitted = true;
    console.log('[Quiz] submit() done, overall =', state.results.overall);
    nextTick(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  }

  function reset() {
    console.log('[Quiz] reset()');
    state.answers = {};
    state.submitted = false;
    state.results = null;
    state.currentSectionIdx = 0;
    nextTick(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  }

  function goSection(idx) {
    if (idx < 0 || idx >= state.sections.length) return;
    console.log('[Quiz] goSection:', idx, state.sections[idx].key);
    state.currentSectionIdx = idx;
    nextTick(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  }

  // ============================================
  // 加载
  // ============================================
  async function loadQuiz() {
    try {
      const resp = await fetch('./quiz.md', { cache: 'no-cache' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status + ' 加载 quiz.md 失败');
      const md = await resp.text();
      if (typeof window.parseQuiz !== 'function') {
        throw new Error('parseQuiz 未定义，请检查 parser.js 是否加载');
      }
      const data = window.parseQuiz(md);
      if (!data.questions || data.questions.length === 0) {
        throw new Error('quiz.md 解析后没有题目，请检查文件内容');
      }
      state.questions = data.questions;
      state.sections = data.sections;
      state.loading = false;
      console.log('[Quiz] loaded', state.questions.length, 'questions in', state.sections.length, 'sections');
    } catch (e) {
      console.error('[Quiz] 加载失败:', e);
      state.parseError = e.message || String(e);
      state.loading = false;
    }
  }

  // ============================================
  // 组件
  // ============================================
  const QuizApp = {
    template: `
      <div v-if="state.loading" class="center-screen">
        <div class="spinner"></div>
        <p style="color: var(--text-muted)">正在加载题目…</p>
      </div>
      <div v-else-if="state.parseError" class="center-screen">
        <div class="error-card">
          <h3>😢 加载失败</h3>
          <p>{{ state.parseError }}</p>
          <p style="margin-top:12px;font-size:13px">
            请确认你已启动本地 HTTP 服务器（如 <code>pnpm dlx http-server . -p 8765</code>），
            并通过 <code>http://127.0.0.1:8765/index.html</code> 访问。
          </p>
        </div>
      </div>
      <template v-else>
        <quiz-header
          :state="state"
          :on-submit="handleSubmit"
          :on-reset="handleReset"
          :on-back="handleBack"
          :sidebar-open="sidebarOpen"
          :on-toggle-sidebar="toggleSidebar" />

        <div class="layout" :class="{ 'with-sidebar': !state.submitted && sidebarVisible }">
          <section-sidebar
            v-if="!state.submitted"
            :state="state"
            :open="sidebarOpen"
            @close="sidebarOpen = false"
            @go="goSection" />

          <div class="layout-content">
            <div v-if="!state.submitted" class="quiz-main">
              <section-view
                :state="state"
                :on-pick="handlePick"
                :on-submit="handleSubmit" />
            </div>
            <results-summary
              v-else
              :state="state"
              :on-reset="handleReset"
              :on-back="handleBack" />
          </div>
        </div>

        <div v-if="!state.submitted && sidebarOpen"
          class="sidebar-backdrop"
          @click="sidebarOpen = false"></div>
      </template>
    `,
    props: ['state'],
    data() {
      return { sidebarOpen: false };
    },
    computed: {
      sidebarVisible() {
        // 桌面端总是显示，移动端根据 sidebarOpen
        return true; // 占位，实际由 CSS 控制
      },
    },
    methods: {
      handleSubmit() { console.log('[QuizApp] handleSubmit called'); submit(); },
      handleReset() { reset(); this.sidebarOpen = false; },
      handleBack() { state.submitted = false; this.sidebarOpen = false; },
      handlePick(qid, key) { pickAnswer(qid, key); },
      toggleSidebar() { this.sidebarOpen = !this.sidebarOpen; },
      goSection(idx) { goSection(idx); this.sidebarOpen = false; },
    },
  };

  const QuizHeader = {
    template: `
      <header class="quiz-header">
        <button v-if="!state.submitted" class="hamburger"
          @click="onToggleSidebar"
          :aria-label="sidebarOpen ? '关闭侧栏' : '打开侧栏'">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="2">
            <line x1="3" y1="6" x2="21" y2="6"></line>
            <line x1="3" y1="12" x2="21" y2="12"></line>
            <line x1="3" y1="18" x2="21" y2="18"></line>
          </svg>
        </button>
        <div class="quiz-logo">
          <div class="quiz-logo-icon">IC</div>
          <div class="quiz-logo-title">
            <span>铁鹰学习</span>
            <small>Iron Condor Quiz</small>
          </div>
        </div>
        <div class="quiz-header-spacer"></div>
        <div v-if="!state.submitted" class="quiz-header-progress">
          <div class="quiz-header-progress-text">
            已答 <strong>{{ answeredCount }}</strong> / {{ state.questions.length }}
          </div>
          <div class="quiz-header-progress-bar">
            <div class="quiz-header-progress-fill"
              :style="{ width: overallProgress + '%' }"></div>
          </div>
        </div>
        <button v-if="!state.submitted" class="btn-primary"
          :disabled="answeredCount === 0"
          @click="onSubmit">提交测试</button>
        <button v-else class="btn-ghost" @click="onBack">← 返回答题</button>
        <button v-if="state.submitted" class="btn-success" @click="onReset">重新开始</button>
      </header>
    `,
    props: ['state', 'onSubmit', 'onReset', 'onBack', 'sidebarOpen', 'onToggleSidebar'],
    computed: {
      answeredCount() {
        return Object.values(this.state.answers).filter(a => a && a.length > 0).length;
      },
      overallProgress() {
        if (!this.state.questions.length) return 0;
        return Math.round((this.answeredCount / this.state.questions.length) * 100);
      },
    },
  };

  // 侧栏组件（替代原 SectionTabs）
  const SectionSidebar = {
    template: `
      <aside class="sidebar" :class="{ open: open }" aria-label="章节导航">
        <div class="sidebar-header">
          <h3>📑 章节</h3>
          <button class="sidebar-close"
            @click="$emit('close')"
            aria-label="关闭侧栏">×</button>
        </div>
        <nav class="sidebar-nav">
          <button
            v-for="(s, idx) in state.sections"
            :key="s.key"
            class="sidebar-item"
            :class="{ active: idx === state.currentSectionIdx }"
            @click="$emit('go', idx)">
            <span class="sidebar-item-dot"
              v-if="state.submitted"
              :class="dotClass(s)"></span>
            <span class="sidebar-item-title">{{ s.title }}</span>
            <span class="sidebar-item-count">{{ s.questionIds.length }}</span>
          </button>
        </nav>
        <div class="sidebar-footer">
          <span class="sidebar-tip">点击章节切换</span>
        </div>
      </aside>
    `,
    props: ['state', 'open'],
    emits: ['close', 'go'],
    methods: {
      dotClass(s) {
        if (!this.state.results) return '';
        const r = this.state.results.perSection.find(x => x.key === s.key);
        if (!r) return '';
        if (r.correct === r.total) return 'green';
        if (r.correct === 0 && r.partial === 0) return 'red';
        if (r.correct > 0 || r.partial > 0) return 'amber';
        return 'red';
      },
    },
  };

  const SectionView = {
    template: `
      <section>
        <div class="section-heading" v-if="currentSection">
          <h2>{{ currentSection.title }}</h2>
          <div class="section-progress">
            <strong>{{ answeredInSection }}</strong> / {{ currentSection.questionIds.length }}
          </div>
        </div>
        <div class="section-intro" v-if="currentSection && currentSection.introHtml"
          v-html="currentSection.introHtml"></div>

        <question-card
          v-for="(q, i) in sectionQuestions"
          :key="q.id"
          :q="q"
          :index="i + 1"
          :picked="(state.answers[q.id] || [])"
          :result="state.results && state.results.perQuestion[q.id]"
          :submitted="state.submitted"
          @pick="(key) => handlePick(q.id, key)" />

        <div class="empty-state" v-if="!currentSection || !sectionQuestions.length">
          本章节暂无题目
        </div>

        <div class="section-nav" v-if="state.sections.length > 1">
          <button class="btn-ghost"
            :disabled="state.currentSectionIdx === 0"
            @click="prev">← 上一章</button>
          <button class="btn-ghost"
            :disabled="state.currentSectionIdx === state.sections.length - 1"
            @click="next">下一章 →</button>
        </div>

        <div class="submit-bar" v-if="totalAnswered > 0">
          <div class="submit-bar-inner">
            <span style="color: var(--text-soft); font-size: 13px; align-self: center; padding: 0 8px;">
              已答 {{ totalAnswered }} / {{ state.questions.length }}
            </span>
            <button class="btn-primary"
              type="button"
              @click="handleSubmit"
              :disabled="totalAnswered === 0">提交测试</button>
          </div>
        </div>
      </section>
    `,
    props: ['state', 'onPick', 'onSubmit'],
    computed: {
      currentSection() {
        return this.state.sections[this.state.currentSectionIdx];
      },
      sectionQuestions() {
        if (!this.currentSection) return [];
        const map = new Map(this.state.questions.map(q => [q.id, q]));
        return this.currentSection.questionIds.map(id => map.get(id)).filter(Boolean);
      },
      answeredInSection() {
        return this.sectionQuestions.filter(q => {
          const a = this.state.answers[q.id];
          return a && a.length > 0;
        }).length;
      },
      totalAnswered() {
        return Object.values(this.state.answers).filter(a => a && a.length > 0).length;
      },
    },
    methods: {
      handlePick(qid, key) { this.onPick && this.onPick(qid, key); },
      handleSubmit() {
        console.log('[SectionView] submit-bar button clicked');
        this.onSubmit && this.onSubmit();
      },
      prev() {
        if (this.state.currentSectionIdx > 0) {
          this.state.currentSectionIdx--;
          nextTick(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
        }
      },
      next() {
        if (this.state.currentSectionIdx < this.state.sections.length - 1) {
          this.state.currentSectionIdx++;
          nextTick(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
        }
      },
    },
  };

  const QuestionCard = {
    template: `
      <article class="question-card"
        :class="cardClass">
        <div class="question-header">
          <span class="question-number">Q{{ index }}</span>
          <span class="question-type" :class="q.type">
            {{ q.type === 'single' ? '单选' : '多选' }}
          </span>
          <span v-if="q.difficulty" class="question-difficulty" :class="q.difficulty">
            {{ diffLabel }}
          </span>
        </div>
        <div class="question-body" v-html="q.questionHtml"></div>
        <div class="question-options">
          <div
            v-for="opt in q.options"
            :key="opt.key"
            class="option"
            :class="optionClass(opt)"
            @click="!submitted && $emit('pick', opt.key)">
            <span class="option-marker" :class="{ checkbox: q.type === 'multiple' }">
              <template v-if="q.type === 'multiple' && isPicked(opt.key)">✓</template>
              <template v-else-if="q.type === 'single'">{{ opt.key }}</template>
              <template v-else>{{ opt.key }}</template>
            </span>
            <span class="option-text">{{ opt.text }}</span>
            <span v-if="submitted && isRight(opt.key) && !isPicked(opt.key)" class="option-badge is-correct">正确答案</span>
            <span v-if="submitted && isPicked(opt.key) && !isRight(opt.key)" class="option-badge is-wrong">你的选择</span>
          </div>
        </div>
        <details v-if="submitted" class="question-explanation" open>
          <summary>
            答案解析
            <span v-if="result && result.partial" style="color: var(--warning); font-weight: 700; margin-left: 4px;">
              (部分正确)
            </span>
            <span v-else-if="result && result.correct" style="color: var(--success); font-weight: 700; margin-left: 4px;">
              ✓ 答对了
            </span>
            <span v-else style="color: var(--danger); font-weight: 700; margin-left: 4px;">
              ✗ 答错了
            </span>
          </summary>
          <div class="question-explanation-body" v-html="q.explanationHtml"></div>
        </details>
      </article>
    `,
    props: ['q', 'index', 'picked', 'result', 'submitted'],
    emits: ['pick'],
    computed: {
      diffLabel() {
        return { easy: '简单', medium: '中等', hard: '困难' }[this.q.difficulty] || '';
      },
      cardClass() {
        if (!this.submitted) return '';
        if (this.result && this.result.correct) return 'correct';
        if (this.result && this.result.partial) return 'partial';
        return 'wrong';
      },
    },
    methods: {
      isPicked(key) { return this.picked.includes(key); },
      isRight(key) { return this.q.answer.includes(key); },
      optionClass(opt) {
        const cls = [];
        if (this.submitted) cls.push('disabled');
        if (this.isPicked(opt.key)) cls.push('selected');
        if (this.submitted) {
          if (this.isRight(opt.key)) cls.push('correct-answer');
          else if (this.isPicked(opt.key)) cls.push('wrong-answer');
        }
        return cls;
      },
    },
  };

  const ResultsSummary = {
    template: `
      <div class="results-page">
        <div class="score-card">
          <div class="score-card-label">总体正确率</div>
          <div class="score-card-value">
            {{ pct }}<sup>%</sup>
          </div>
          <div class="score-card-meta">
            <strong>{{ state.results.overall.correct }}</strong>
            / {{ state.results.overall.total }} 题答对
            <span style="opacity:0.7"> ·  </span>
            <strong>{{ Math.round(state.results.overall.pointsGot * 10) / 10 }}</strong>
            / {{ state.results.overall.points }} 分
            <span v-if="state.results.overall.answered < state.results.overall.total" style="opacity:0.7">
              （未答 {{ state.results.overall.total - state.results.overall.answered }} 题）
            </span>
          </div>
        </div>

        <div class="results-section-list">
          <h3>📊 各章节正确率</h3>
          <div v-for="r in state.results.perSection" :key="r.key" class="section-result">
            <div class="section-result-name">{{ r.title }}</div>
            <div class="section-result-bar">
              <div class="section-result-bar-fill"
                :class="barClass(r)"
                :style="{ width: Math.round(r.rate * 100) + '%' }"></div>
            </div>
            <div class="section-result-stat">
              {{ r.correct }} / {{ r.total }}
              <span class="pct" :class="pctClass(r)">{{ Math.round(r.rate * 100) }}%</span>
            </div>
          </div>
        </div>

        <div v-if="wrongQuestions.length > 0">
          <h3 class="wrong-list-title">
            错题 / 部分正确 列表
            <span class="count">{{ wrongQuestions.length }}</span>
          </h3>
          <div v-for="(group, key) in wrongBySection" :key="key">
            <div class="wrong-section-title">{{ group.title }}</div>
            <question-card
              v-for="(q, i) in group.questions"
              :key="q.id"
              :q="q"
              :index="i + 1"
              :picked="state.answers[q.id] || []"
              :result="state.results.perQuestion[q.id]"
              :submitted="true" />
          </div>
        </div>

        <div v-else class="results-section-list" style="text-align:center; padding: 40px;">
          <h3 style="margin: 0 0 4px;">🎉 全部答对！</h3>
          <p style="color: var(--text-muted); margin: 0;">你对铁鹰的理解已经很扎实。</p>
        </div>

        <div style="text-align:center; margin-top: 32px;">
          <button class="btn-success" @click="onReset">🔄 重新开始</button>
        </div>
      </div>
    `,
    props: ['state', 'onReset', 'onBack'],
    computed: {
      pct() {
        return Math.round(this.state.results.overall.rate * 100);
      },
      wrongQuestions() {
        return this.state.questions.filter(q => {
          const r = this.state.results.perQuestion[q.id];
          return r && !r.correct;
        });
      },
      wrongBySection() {
        const map = {};
        for (const q of this.wrongQuestions) {
          if (!map[q.section]) {
            const sec = this.state.sections.find(s => s.key === q.section);
            map[q.section] = {
              title: sec ? sec.title : q.section,
              questions: [],
            };
          }
          map[q.section].questions.push(q);
        }
        return map;
      },
    },
    methods: {
      barClass(r) {
        if (r.rate >= 0.85) return 'perfect';
        if (r.rate >= 0.5) return 'partial';
        return 'low';
      },
      pctClass(r) {
        if (r.rate >= 0.85) return 'high';
        if (r.rate < 0.5) return 'low';
        return '';
      },
    },
  };

  // ============================================
  // 注册 + 挂载
  // ============================================
  const app = createApp({
    setup() {
      onMounted(loadQuiz);
      return { state };
    },
    template: `<quiz-app :state="state" />`,
  });

  app.component('quiz-app', QuizApp);
  app.component('quiz-header', QuizHeader);
  app.component('section-sidebar', SectionSidebar);
  app.component('section-view', SectionView);
  app.component('question-card', QuestionCard);
  app.component('results-summary', ResultsSummary);

  app.mount('#app');

  // 调试用：暴露到 window
  window.__quiz = { state, submit, reset, pickAnswer, goSection };
})();
