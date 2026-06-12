// progress.js — localStorage 进度持久化
// 配合 progress.config.js 中的 PROGRESS_SCHEMA_VERSION 做 schema 迁移
//
// 用法：
//   const progress = loadProgress();
//   saveProgress(progress);

const { STORAGE_KEYS, PROGRESS_SCHEMA_VERSION } = window.PROGRESS_CONFIG;

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
    chapters: {},          // ch00: { status, opened_at, completed_at, time_spent_sec, scroll_percent, quiz: { qid: {...} } }
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
  // 老数据补全字段
  if (p.version !== PROGRESS_SCHEMA_VERSION) {
    console.info(`progress schema ${p.version} → ${PROGRESS_SCHEMA_VERSION}`);
    p.version = PROGRESS_SCHEMA_VERSION;
  }
  // 迁移：把老 key（无 :lang 后缀）重命名为 `${qid}:${lang}`
  // 用 answer.lang 决定后缀，没有 lang 字段则按当前 loadLang 推断
  const defaultLang = (typeof localStorage !== "undefined"
    && localStorage.getItem(STORAGE_KEYS.LANG)) || "zh";
  for (const ch of Object.values(p.chapters)) {
    if (!ch.quiz) continue;
    const oldKeys = Object.keys(ch.quiz).filter(
      (k) => !k.endsWith(":zh") && !k.endsWith(":en")
    );
    if (oldKeys.length === 0) continue;
    const newQuiz = {};
    for (const [k, v] of Object.entries(ch.quiz)) {
      if (k.endsWith(":zh") || k.endsWith(":en")) {
        newQuiz[k] = v;
      } else {
        const lang = v.lang || defaultLang;
        newQuiz[`${k}:${lang}`] = v;
      }
    }
    ch.quiz = newQuiz;
  }
  return p;
}

export function getChapterProgress(progress, chapterId) {
  if (!progress.chapters[chapterId]) {
    progress.chapters[chapterId] = {
      status: "unlocked",
      opened_at: null,
      completed_at: null,
      time_spent_sec: 0,
      scroll_percent: 0,
      quiz: {},
    };
  }
  return progress.chapters[chapterId];
}

export function setQuizAnswer(progress, chapterId, qid, answerObj) {
  const ch = getChapterProgress(progress, chapterId);
  ch.quiz[qid] = { ...answerObj, answered_at: new Date().toISOString() };
  saveProgress(progress);
}

export function markChapterOpened(progress, chapterId) {
  const ch = getChapterProgress(progress, chapterId);
  if (!ch.opened_at) ch.opened_at = new Date().toISOString();
  if (ch.status === "unlocked" || ch.status === "locked") ch.status = "in_progress";
  ch.scroll_percent = Math.max(ch.scroll_percent, 0);
  progress.last_visited_at = new Date().toISOString();
  saveProgress(progress);
}

export function markChapterCompleted(progress, chapterId) {
  const ch = getChapterProgress(progress, chapterId);
  ch.status = "completed";
  ch.completed_at = new Date().toISOString();
  saveProgress(progress);
}

export function updateScrollPercent(progress, chapterId, pct) {
  const ch = getChapterProgress(progress, chapterId);
  ch.scroll_percent = Math.max(ch.scroll_percent, pct);
  ch.time_spent_sec += 1;
  saveProgress(progress);
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
// language: 'zh' | 'en' | undefined
//   - 传 language 时，只统计该语言的答题（中英分开计数）
//   - 不传则统计全部（向后兼容 / 调试用）
export function calcSummary(progress, chaptersMeta, language) {
  const completedChapters = Object.values(progress.chapters).filter(
    (c) => c.status === "completed"
  ).length;
  const totalChapters = chaptersMeta.length;
  let quizzesAnswered = 0;
  let quizzesCorrect = 0;
  let quizzesTotal = 0;
  for (const meta of chaptersMeta) {
    // 总题数也按语言切分：当前语言版本的章节元数据对应的题数
    // 由于 quiz_count 同一章 zh/en 数量已对齐（手动保证），直接累加即可
    quizzesTotal += meta.quiz_count || 0;
    const ch = progress.chapters[meta.id];
    if (!ch) continue;
    for (const q of Object.values(ch.quiz || {})) {
      // 按语言过滤：未指定 language 时不筛
      if (language && q.lang && q.lang !== language) continue;
      quizzesAnswered++;
      if (q.correct === true) quizzesCorrect++;
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
