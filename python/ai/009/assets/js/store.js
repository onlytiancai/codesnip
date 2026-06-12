// store.js — 迷你 store（reactive + provide/inject）
// 替代 Pinia（Pinia 无 IIFE 构建，必须 importmap）
import { reactive, readonly, provide, inject } from "vue";
import {
  loadProgress, saveProgress,
  loadTheme, saveTheme, loadLang, saveLang,
  loadStudentName, saveStudentName,
  loadCert, saveCert,
  getChapterProgress, setQuizAnswer, markChapterOpened,
  markChapterCompleted, updateScrollPercent,
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
}

export function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
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
  },
  setStudentName(name) {
    state.studentName = name;
    saveStudentName(name);
  },
  openChapter(chapterId) {
    state.currentChapterId = chapterId;
    markChapterOpened(state.progress, chapterId);
  },
  answerQuiz(chapterId, qid, answerObj) {
    setQuizAnswer(state.progress, chapterId, qid, answerObj);
  },
  scrollChapter(chapterId, pct) {
    updateScrollPercent(state.progress, chapterId, pct);
  },
  completeChapter(chapterId) {
    markChapterCompleted(state.progress, chapterId);
  },
  issueCert(certObj) {
    saveCert(certObj);
  },
  getCert() { return loadCert(); },
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
