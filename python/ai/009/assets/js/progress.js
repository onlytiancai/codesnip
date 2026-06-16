// progress.js — localStorage 进度持久化
// 配合 progress.config.js 中的 PROGRESS_SCHEMA_VERSION 做 schema 迁移
//
// 用法：
//   const progress = loadProgress();
//   saveProgress(progress);
//
// 数据结构（schema v3）：
//   progress.chapters[chapterId] = {
//     zh: { status, opened_at, completed_at, time_spent_sec, scroll_percent },
//     en: { status, opened_at, completed_at, time_spent_sec, scroll_percent },
//     quiz: { "q1-1:zh": {...}, "q1-1:en": {...} },   // key 带 :lang 后缀
//   }
//   章节状态严格按语言拆分；切换语言互不影响

const { STORAGE_KEYS, PROGRESS_SCHEMA_VERSION } = window.PROGRESS_CONFIG;

const LANG_DEFAULTS = Object.freeze({
  status: "unlocked",
  opened_at: null,
  completed_at: null,
  time_spent_sec: 0,
  scroll_percent: 0,
});

export function loadProgress() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.PROGRESS);
    if (!raw) return defaultProgress();
    const p = JSON.parse(raw);
    return normalizeProgress(p);
  } catch (e) {
    console.warn("loadProgress failed, returning default:", e);
    return defaultProgress();
  }
}

export function saveProgress(progress) {
  try {
    localStorage.setItem(STORAGE_KEYS.PROGRESS, JSON.stringify(progress));
  } catch (e) {
    console.error("saveProgress failed:", e);
  }
}

export function defaultProgress() {
  return {
    version: PROGRESS_SCHEMA_VERSION,
    started_at: new Date().toISOString(),
    last_visited_at: new Date().toISOString(),
    chapters: {},
    summary: {
      chapters_completed: 0,
      chapters_total: 0,
      quizzes_answered: 0,
      quizzes_correct: 0,
      quizzes_total: 0,
    },
  };
}

export function normalizeProgress(p) {
  if (!p) p = {};
  if (!p.version) p.version = 1;
  if (!p.chapters) p.chapters = {};
  if (!p.summary) p.summary = {};
  if (p.version !== PROGRESS_SCHEMA_VERSION) {
    console.info(`progress schema ${p.version} → ${PROGRESS_SCHEMA_VERSION}`);
    p.version = PROGRESS_SCHEMA_VERSION;
  }

  const defaultLang = (typeof localStorage !== "undefined"
    && localStorage.getItem(STORAGE_KEYS.LANG)) || "zh";

  for (const ch of Object.values(p.chapters)) {
    // 1) 老 chapter 字段（status/opened_at/.../ 单值）→ ch[defaultLang]
    if (ch.status !== undefined && typeof ch.status === "string") {
      ch[defaultLang] = {
        status: ch.status,
        opened_at: ch.opened_at ?? null,
        completed_at: ch.completed_at ?? null,
        time_spent_sec: ch.time_spent_sec || 0,
        scroll_percent: ch.scroll_percent || 0,
      };
      ch.en = { ...LANG_DEFAULTS };
      delete ch.status;
      delete ch.opened_at;
      delete ch.completed_at;
      delete ch.time_spent_sec;
      delete ch.scroll_percent;
    }
    if (!ch.zh) ch.zh = { ...LANG_DEFAULTS };
    if (!ch.en) ch.en = { ...LANG_DEFAULTS };
    // 补齐每个语言可能缺失的字段
    for (const lang of ["zh", "en"]) {
      for (const k of Object.keys(LANG_DEFAULTS)) {
        if (ch[lang][k] === undefined) ch[lang][k] = LANG_DEFAULTS[k];
      }
    }

    // 2) 老 quiz key（无 :lang 后缀）→ `${qid}:${lang}`
    if (!ch.quiz) ch.quiz = {};
    const oldKeys = Object.keys(ch.quiz).filter(
      (k) => !k.endsWith(":zh") && !k.endsWith(":en")
    );
    if (oldKeys.length > 0) {
      const newQuiz = {};
      for (const [k, v] of Object.entries(ch.quiz)) {
        if (k.endsWith(":zh") || k.endsWith(":en")) {
          newQuiz[k] = v;
        } else {
          newQuiz[`${k}:${v.lang || defaultLang}`] = v;
        }
      }
      ch.quiz = newQuiz;
    }
  }
  return p;
}

export function getChapterProgress(progress, chapterId) {
  if (!progress.chapters[chapterId]) {
    progress.chapters[chapterId] = {
      zh: { ...LANG_DEFAULTS },
      en: { ...LANG_DEFAULTS },
      quiz: {},
    };
  }
  return progress.chapters[chapterId];
}

// 获取/创建指定语言的章节子对象
function getLangState(progress, chapterId, lang) {
  const ch = getChapterProgress(progress, chapterId);
  if (!ch[lang]) ch[lang] = { ...LANG_DEFAULTS };
  for (const k of Object.keys(LANG_DEFAULTS)) {
    if (ch[lang][k] === undefined) ch[lang][k] = LANG_DEFAULTS[k];
  }
  return ch[lang];
}

export function setQuizAnswer(progress, chapterId, qid, answerObj) {
  const ch = getChapterProgress(progress, chapterId);
  ch.quiz[qid] = { ...answerObj, answered_at: new Date().toISOString() };
  saveProgress(progress);
}

export function markChapterOpened(progress, chapterId, lang) {
  const ls = getLangState(progress, chapterId, lang);
  if (!ls.opened_at) ls.opened_at = new Date().toISOString();
  if (ls.status === "unlocked" || ls.status === "locked") ls.status = "in_progress";
  ls.scroll_percent = Math.max(ls.scroll_percent, 0);
  progress.last_visited_at = new Date().toISOString();
  saveProgress(progress);
}

export function markChapterCompleted(progress, chapterId, lang) {
  const ls = getLangState(progress, chapterId, lang);
  ls.status = "completed";
  ls.completed_at = new Date().toISOString();
  saveProgress(progress);
}

export function updateScrollPercent(progress, chapterId, lang, pct) {
  const ls = getLangState(progress, chapterId, lang);
  ls.scroll_percent = Math.max(ls.scroll_percent, pct);
  ls.time_spent_sec += 1;
  saveProgress(progress);
}

export function getStatus(progress, chapterId, lang) {
  return getLangState(progress, chapterId, lang).status;
}

export function getScrollPercent(progress, chapterId, lang) {
  return getLangState(progress, chapterId, lang).scroll_percent;
}

// ====== 主题/语言/姓名 ======
export function loadTheme() {
  return localStorage.getItem(STORAGE_KEYS.THEME) || "light";
}
export function saveTheme(theme) {
  localStorage.setItem(STORAGE_KEYS.THEME, theme);
}
export function loadLang() {
  return localStorage.getItem(STORAGE_KEYS.LANG) || "zh";
}
export function saveLang(lang) {
  localStorage.setItem(STORAGE_KEYS.LANG, lang);
}
export function loadStudentName() {
  return localStorage.getItem(STORAGE_KEYS.STUDENT_NAME) || "";
}
export function saveStudentName(name) {
  localStorage.setItem(STORAGE_KEYS.STUDENT_NAME, name);
}

// ====== 证书 ======
export function loadCert() {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.CERT);
    return raw ? JSON.parse(raw) : null;
  } catch (e) { return null; }
}
export function saveCert(cert) {
  localStorage.setItem(STORAGE_KEYS.CERT, JSON.stringify(cert));
}

// ====== 汇总统计（实时计算）======
// language: 'zh' | 'en'
//   - 完成章节数：当前语言下 status === "completed"
//   - 准确率：当前语言的答题 + 简答按"参与 = 正确"计 1 分
export function calcSummary(progress, chaptersMeta, language) {
  // 完成章节数：按当前语言 ch[lang].status 统计
  const completedChapters = chaptersMeta.filter((meta) => {
    const ch = progress.chapters[meta.id];
    return ch && ch[language]?.status === "completed";
  }).length;
  const totalChapters = chaptersMeta.length;

  let quizzesAnswered = 0;
  let quizzesCorrect = 0;
  let quizzesTotal = 0;
  for (const meta of chaptersMeta) {
    quizzesTotal += meta.quiz_count || 0;
    const ch = progress.chapters[meta.id];
    if (!ch) continue;
    for (const q of Object.values(ch.quiz || {})) {
      // 只统计当前语言的答题
      if (q.lang && q.lang !== language) continue;
      quizzesAnswered++;
      // 单选/多选答对 + 简答（correct: null）都计 1 分
      if (q.correct === true || q.correct === null) quizzesCorrect++;
    }
  }
  return {
    chapters_completed: completedChapters,
    chapters_total: totalChapters,
    quizzes_answered: quizzesAnswered,
    quizzes_correct: quizzesCorrect,
    quizzes_total: quizzesTotal,
    overall_pct: totalChapters === 0 ? 0 : completedChapters / totalChapters,
    quiz_pct: quizzesTotal === 0 ? 0 : quizzesCorrect / quizzesTotal,
  };
}

// ====== 重置（清空进度、证书、姓名；保留主题和语言偏好）======
export function resetAllProgress() {
  try {
    localStorage.removeItem(STORAGE_KEYS.PROGRESS);
    localStorage.removeItem(STORAGE_KEYS.CERT);
    localStorage.removeItem(STORAGE_KEYS.STUDENT_NAME);
  } catch (e) {
    console.error("resetAllProgress failed:", e);
  }
}
