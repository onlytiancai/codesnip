// ChapterView.js — 章节详情
// 1. 根据路由 :id 找章节元数据
// 2. fetch 对应语言的 markdown
// 3. marked 解析 + KaTeX 渲染
// 4. 扫描 [data-block-type] 占位 div，mount 对应 Vue 组件
// 5. 滚动到底/答题进度写入 store

import { ref, onMounted, onBeforeUnmount, watch, nextTick, createApp, h } from "vue";
import { useStore, provideStore } from "../store.js";
import { renderMarkdown, parseQuizBlock, parseChartBlock,
         parseGraphBlock, parseNetworkBlock, parseTrainDemoBlock, parseFormulaBlock } from "../markdown.js";
import { Quiz } from "./Quiz.js";
import { ChartBlock } from "./ChartBlock.js";
import { ComputeGraph } from "./ComputeGraph.js";
import { NetworkViz } from "./NetworkViz.js";
import { TrainDemo } from "./TrainDemo.js";
import { Formula } from "./Formula.js";
import { I18N, pick } from "../i18n.js";

const COMP_MAP = {
  "quiz": Quiz,
  "chart": ChartBlock,
  "graph": ComputeGraph,
  "network": NetworkViz,
  "train-demo": TrainDemo,
  "formula": Formula,
};

export const ChapterView = {
  props: ["id"],
  setup(props) {
    const { state, actions } = useStore();
    const mdHtml = ref("");
    const pendingBlocks = ref([]);
    const loading = ref(false);
    const error = ref(null);
    let mountedApps = [];   // 跟踪挂载的子 app，卸载时 dispose
    let katexTimer = null;

    function t(key) { return pick(I18N.ui[key] || { zh: key, en: key }, state.language); }

    function meta() { return state.chaptersMap[props.id]; }
    function isCompleted(id) {
      return state.progress?.chapters?.[id]?.status === "completed";
    }
    function isLocked(id) {
      const m = state.chaptersMap[id];
      if (!m?.prerequisites?.length) return false;
      return m.prerequisites.some((p) => !isCompleted(p));
    }

    function getAdjacent() {
      const idx = state.chapters.findIndex((c) => c.id === props.id);
      return {
        prev: idx > 0 ? state.chapters[idx - 1] : null,
        next: idx < state.chapters.length - 1 ? state.chapters[idx + 1] : null,
      };
    }

    async function loadChapter() {
      const m = meta();
      if (!m) { error.value = "Chapter not found: " + props.id; return; }
      if (isLocked(m.id)) { error.value = "locked"; return; }

      loading.value = true;
      error.value = null;
      try {
        // 清理之前的子 app
        unmountBlocks();

        const path = state.language === "en" ? m.file_en : m.file_zh;
        const res = await fetch(path);
        if (!res.ok) throw new Error(`Failed to load: ${res.status}`);
        const text = await res.text();
        const result = await renderMarkdown(text);
        mdHtml.value = result.html;
        pendingBlocks.value = result.blocks;
        actions.openChapter(m.id);

        // 等两轮 microtask + 一个 rAF，确保 router-view 切完 + v-html 渲染完成
        await nextTick();
        await new Promise(r => requestAnimationFrame(r));
        mountBlocks();
        renderMath();
        setupScrollTracking();
      } catch (e) {
        error.value = e.message;
        console.error(e);
      } finally {
        loading.value = false;
      }
    }

    function mountBlocks(attempt = 0) {
      const root = document.getElementById("chapter-content");
      if (!root) {
        if (attempt < 10) {
          // router-view 切换还没把 #chapter-content 加到 DOM，重试
          setTimeout(() => mountBlocks(attempt + 1), 30);
        } else {
          console.warn("[mountBlocks] no #chapter-content after 10 retries");
        }
        return;
      }
      const placeholders = root.querySelectorAll("[data-block-type]");
      const blocks = pendingBlocks.value;
      placeholders.forEach((el) => {
        const blockId = parseInt(el.dataset.blockId || "0", 10);
        const meta = blocks[blockId];
        if (!meta) {
          console.warn("Missing block meta for id", blockId);
          return;
        }
        const type = meta.type;
        const Comp = COMP_MAP[type];
        if (!Comp) { console.warn("Unknown block type:", type); return; }
        const args = meta.args;
        const body = meta.body;

        // 解析参数
        let parsed;
        if (type === "quiz") parsed = parseQuizBlock(args, body);
        else if (type === "chart") parsed = parseChartBlock(args, body);
        else if (type === "graph") parsed = parseGraphBlock(args, body);
        else if (type === "network") parsed = parseNetworkBlock(args, body);
        else if (type === "train-demo") parsed = parseTrainDemoBlock(args, body);
        else if (type === "formula") parsed = parseFormulaBlock(args, body);

        // 用 Vue 创建子应用挂载到占位 div
        const subApp = createApp({ render: () => h(Comp, { data: parsed, chapterId: props.id }) });
        provideStore(subApp);  // 子 app 也注入同一个 store
        subApp.mount(el);
        mountedApps.push(subApp);
      });
      pendingBlocks.value = [];
    }

    function unmountBlocks() {
      mountedApps.forEach((app) => { try { app.unmount(); } catch (e) {} });
      mountedApps = [];
    }

    function renderMath() {
      if (typeof window.renderMathInElement !== "function") {
        // auto-render 还没加载完
        if (katexTimer) clearTimeout(katexTimer);
        katexTimer = setTimeout(renderMath, 50);
        return;
      }
      const root = document.getElementById("chapter-content");
      if (!root) return;
      try {
        window.renderMathInElement(root, {
          delimiters: [
            { left: "$$", right: "$$", display: true },
            { left: "$", right: "$", display: false },
            { left: "\\[", right: "\\]", display: true },
            { left: "\\(", right: "\\)", display: false },
          ],
          throwOnError: false,
        });
      } catch (e) { console.warn("KaTeX render failed:", e); }
    }

    function setupScrollTracking() {
      const root = document.getElementById("chapter-content");
      if (!root) return;
      const m = meta();
      let lastPct = 0;
      const onScroll = () => {
        const scrollTop = window.scrollY || document.documentElement.scrollTop;
        const max = (document.documentElement.scrollHeight - window.innerHeight) || 1;
        const pct = Math.min(100, Math.round((scrollTop / max) * 100));
        if (pct - lastPct >= 5) {
          lastPct = pct;
          actions.scrollChapter(m.id, pct);
          // 80% 滚动 + 60% 答题正确率 = 完成
          if (pct >= 80) checkCompletion();
        }
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      // 记住 listener 以便清理
      ChapterView._scrollListener = onScroll;
    }

    function checkCompletion() {
      const m = meta();
      const ch = state.progress.chapters[m.id];
      if (!ch || ch.status === "completed") return;
      const answered = Object.values(ch.quiz || {});
      if (answered.length === 0) return;
      const correct = answered.filter((q) => q.correct === true).length;
      const rate = correct / Math.max(1, answered.length);
      if (rate >= 0.6) {
        actions.completeChapter(m.id);
        showToast(t("chapterCompleted") + " 🎉");
      }
    }

    function showToast(msg) {
      const el = document.createElement("div");
      el.className = "toast";
      el.textContent = msg;
      document.body.appendChild(el);
      setTimeout(() => el.remove(), 2000);
    }

    onMounted(loadChapter);
    onBeforeUnmount(() => {
      unmountBlocks();
      if (ChapterView._scrollListener) {
        window.removeEventListener("scroll", ChapterView._scrollListener);
        ChapterView._scrollListener = null;
      }
      if (katexTimer) clearTimeout(katexTimer);
    });

    watch(() => props.id, loadChapter);
    watch(() => state.language, loadChapter);

    return { state, mdHtml, loading, error, meta, getAdjacent, t };
  },
  template: `
    <div v-if="loading" style="text-align:center; padding:60px 20px; color:var(--muted)">
      <div style="font-size:32px">⏳</div>
      <p>Loading...</p>
    </div>

    <div v-else-if="error === 'locked'" class="cert-locked">
      <div class="lock-icon">🔒</div>
      <h2>{{ state.language === 'en' ? 'Chapter Locked' : '本章未解锁' }}</h2>
      <p>{{ state.language === 'en' ? 'Complete the prerequisites first.' : '请先完成前置章节。' }}</p>
    </div>

    <div v-else-if="error" class="cert-locked">
      <div class="lock-icon">😢</div>
      <h2>{{ state.language === 'en' ? 'Failed to load' : '加载失败' }}</h2>
      <p>{{ error }}</p>
    </div>

    <div v-else-if="meta()">
      <div class="chapter-header">
        <div class="chapter-num">CH {{ String(meta().order).padStart(2, '0') }}</div>
        <h1>{{ state.language === 'en' ? meta().title_en : meta().title_zh }}</h1>
        <div class="chapter-summary">
          {{ state.language === 'en' ? meta().summary_en : meta().summary_zh }}
        </div>
      </div>

      <div id="chapter-content" v-html="mdHtml"></div>

      <div class="chapter-nav">
        <button v-if="getAdjacent().prev" @click="$router.push('/ch/' + getAdjacent().prev.id)">
          ← {{ state.language === 'en' ? getAdjacent().prev.title_en : getAdjacent().prev.title_zh }}
        </button>
        <div v-else></div>

        <button v-if="getAdjacent().next" class="primary" @click="$router.push('/ch/' + getAdjacent().next.id)">
          {{ state.language === 'en' ? getAdjacent().next.title_en : getAdjacent().next.title_zh }} →
        </button>
        <router-link v-else to="/cert" class="primary" style="text-decoration:none; display:inline-block">
          🎓 {{ t('cert') || '证书' }}
        </router-link>
      </div>
    </div>
  `,
};
