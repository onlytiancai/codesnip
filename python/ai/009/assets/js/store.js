// store.js — 迷你 store（reactive + provide/inject）
// 替代 Pinia（Pinia 无 IIFE 构建，必须 importmap）
import { reactive, readonly, provide, inject } from "vue";
import {
  loadProgress, saveProgress,
  loadTheme, saveTheme, loadLang, saveLang,
  loadStudentName, saveStudentName,
  loadCert, saveCert,
  setQuizAnswer, markChapterOpened,
  markChapterCompleted, updateScrollPercent, calcSummary,
  resetAllProgress,
} from "./progress.js";

const state = reactive({
  // 静态数据
  chapters: [],          // 从 chapters.json 加载的章节元数据
  chaptersMap: {},       // id -> meta
  certMeta: null,        // 证书配置

  // 运行时状态
  ready: false,          // 数据加载完成
  language: "zh",
  theme: "light",
  currentChapterId: "ch00",
  studentName: "",
  progress: null,        // 见 progress.js 的 schema

  // 路由状态（用于章节导航的"上一章/下一章"）
  prevChapterId: null,
  nextChapterId: null,
});

export function createStore(chapters, certMeta) {
  state.chapters = chapters;
  state.certMeta = certMeta;
  state.chaptersMap = Object.fromEntries(chapters.map((c) => [c.id, c]));

  // 从 localStorage 恢复
  state.theme = loadTheme();
  state.language = loadLang();
  state.studentName = loadStudentName();
  state.progress = loadProgress();
  state.ready = true;
  applyTheme(state.theme);

  // 诊断日志：用户可在 DevTools 控制台查看数据结构和版本
  console.log("[009] store ready, language:", state.language);
  console.log("[009] progress chapters keys:", Object.keys(state.progress.chapters));
  console.log("[009] first chapter sample:", JSON.parse(JSON.stringify(
    Object.values(state.progress.chapters)[0] || null
  )));
}

export function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  // 同步切换 highlight.js 主题
  const hljsLink = document.getElementById("hljs-theme");
  if (hljsLink) {
    hljsLink.href = theme === "dark"
      ? "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github-dark.min.css"
      : "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/github.min.css";
  }
}

export function getState() { return state; }
export function getReadonlyState() { return readonly(state); }

// ====== Actions ======
export const actions = {
  setTheme(theme) {
    state.theme = theme;
    saveTheme(theme);
    applyTheme(theme);
  },
  toggleTheme() {
    const next = state.theme === "light" ? "dark" : "light";
    actions.setTheme(next);
  },
  setLanguage(lang) {
    state.language = lang;
    saveLang(lang);
    // summary 是缓存值，切换语言后必须重算（之前切到英文仍显示中文的统计）
    if (state.progress) {
      state.progress.summary = calcSummary(state.progress, state.chapters, lang);
      // 诊断：每次切语言后打印当前的 chapter 状态分布
      const dist = { zh: { done: 0, inprog: 0 }, en: { done: 0, inprog: 0 } };
      for (const ch of Object.values(state.progress.chapters)) {
        for (const L of ["zh", "en"]) {
          const s = ch[L]?.status;
          if (s === "completed") dist[L].done++;
          else if (s === "in_progress") dist[L].inprog++;
        }
      }
      console.log(`[009] setLanguage(${lang}) → summary=`, JSON.parse(JSON.stringify(state.progress.summary)), "dist=", dist);
    }
  },
  setStudentName(name) {
    state.studentName = name;
    saveStudentName(name);
  },
  openChapter(chapterId) {
    state.currentChapterId = chapterId;
    // 章节打开也按语言记录
    markChapterOpened(state.progress, chapterId, state.language);
  },
  answerQuiz(chapterId, qid, answerObj) {
    setQuizAnswer(state.progress, chapterId, qid, answerObj);
    // 按当前语言统计（quiz.js 传入的 key 已带 :lang 后缀，summary 也按语言分）
    state.progress.summary = calcSummary(state.progress, state.chapters, state.language);
  },
  scrollChapter(chapterId, pct) {
    updateScrollPercent(state.progress, chapterId, state.language, pct);
  },
  completeChapter(chapterId) {
    markChapterCompleted(state.progress, chapterId, state.language);
    state.progress.summary = calcSummary(state.progress, state.chapters, state.language);
  },
  issueCert(certObj) {
    saveCert(certObj);
  },
  getCert() { return loadCert(); },
  // 重置进度：清空 PROGRESS/CERT/STUDENT_NAME，保留主题和语言偏好，然后 reload
  resetAll() {
    resetAllProgress();
    window.location.reload();
  },
};

// ====== Provide/Inject 桥 ======
export const STORE_KEY = Symbol("009-store");

export function provideStore(app) {
  app.provide(STORE_KEY, { state: getReadonlyState(), actions });
}

export function useStore() {
  const ctx = inject(STORE_KEY);
  if (!ctx) throw new Error("useStore() called without provideStore(app)");
  return ctx;
}
