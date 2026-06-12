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
        // 按当前语言切换图片（要求 assets/images/ 下的图都有 _zh/_en 双版本）
        const langSuffix = state.language === 'en' ? '_en' : '_zh';
        const localizedHtml = result.html.replace(
          /(<img[^>]+src=["'])([^"']*?\/([^/'" ]+))\.png(["'])/g,
          (m, pre, _path, name, post) => {
            if (!_path.includes('assets/images/')) return m;
            if (/_zh$/.test(name) || /_en$/.test(name)) return m;
            return `${pre}${_path}${langSuffix}.png${post}`;
          }
        );
        mdHtml.value = localizedHtml;
        pendingBlocks.value = result.blocks;
        actions.openChapter(m.id);
        // 用户回头看章节时，若之前已满足条件则补标记完成
        checkCompletion();

        // 等两轮 microtask + 一个 rAF，确保 router-view 切完 + v-html 渲染完成
        await nextTick();
        await new Promise(r => requestAnimationFrame(r));
        mountBlocks();
        renderMath();
        highlightCode();
        setupScrollTracking();
      } catch (e) {
        error.value = e.message;
        console.error(e);
      } finally {
        loading.value = false;
      }
    }

    // 对 body 里的图片路径加语言后缀（支持 markdown 格式 ![alt](url) 和 HTML <img src="...">）
    function localizeImagesInBody(body) {
      if (!body) return body;
      const suffix = state.language === 'en' ? '_en' : '_zh';
      // 1) markdown 格式：![alt](path/xxx.png)
      let out = body.replace(
        /(!\[[^\]]*\]\()([^)]*?\/([^/)]+))\.png(\))/g,
        (m, pre, path, name, post) => {
          if (!path.includes('assets/images/')) return m;
          // 跳过已带 _zh / _en 后缀的（ch01/ch05 旧 markdown 写死）
          if (/_zh$/.test(name) || /_en$/.test(name)) return m;
          return `${pre}${path}${suffix}.png${post}`;
        }
      );
      // 2) HTML 格式：<img src="path/xxx.png">（防止未来扩展）
      out = out.replace(
        /(<img[^>]+src=["'])([^"']*?\/([^/'" ]+))\.png(["'])/g,
        (m, pre, path, name, post) => {
          if (!path.includes('assets/images/')) return m;
          if (/_zh$/.test(name) || /_en$/.test(name)) return m;
          return `${pre}${path}${suffix}.png${post}`;
        }
      );
      return out;
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
        try {
          // 对 body 跑一次图片语言后缀替换（::: 块在 markdown 阶段被替换成占位 div，
          // img 标签仍在 body 里没经过 result.html 的后缀处理）
          const localizedBody = localizeImagesInBody(body);
          if (type === "quiz") parsed = parseQuizBlock(args, localizedBody);
          else if (type === "chart") parsed = parseChartBlock(args, localizedBody);
          else if (type === "graph") parsed = parseGraphBlock(args, localizedBody);
          else if (type === "network") parsed = parseNetworkBlock(args, localizedBody);
          else if (type === "train-demo") parsed = parseTrainDemoBlock(args, localizedBody);
          else if (type === "formula") parsed = parseFormulaBlock(args, localizedBody);
        } catch (e) {
          console.error(`[mountBlocks] parse error for block ${blockId} (${type}):`, e);
          return;
        }

        // 用 Vue 创建子应用挂载到占位 div（单块挂载失败不影响其他块）
        try {
          const subApp = createApp({ render: () => h(Comp, { data: parsed, chapterId: props.id }) });
          provideStore(subApp);  // 子 app 也注入同一个 store
          subApp.mount(el);
          mountedApps.push(subApp);
        } catch (e) {
          console.error(`[mountBlocks] mount error for block ${blockId} (${type}):`, e);
        }
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
          strict: "ignore",  // \text{} 里出现 → 等未识别 Unicode 时不再 warn
        });
      } catch (e) { console.warn("KaTeX render failed:", e); }
    }

    // 高亮代码块（必须在 DOMPurify sanitize 之后、且 hljs 已加载）
    // hljs 是 UMD CDN，可能比 v-html 晚几个 tick；未加载则轮询重试
    let hljsTimer = null;
    function highlightCode() {
      if (typeof window.hljs === "undefined") {
        if (hljsTimer) clearTimeout(hljsTimer);
        hljsTimer = setTimeout(highlightCode, 50);
        return;
      }
      const root = document.getElementById("chapter-content");
      if (!root) {
        // 章节内容尚未渲染（router-view 异步切换）
        if (hljsTimer) clearTimeout(hljsTimer);
        hljsTimer = setTimeout(highlightCode, 100);
        return;
      }
      try {
        root.querySelectorAll("pre code").forEach((el) => {
          if (el.classList && el.classList.contains("hljs")) return;
          window.hljs.highlightElement(el);
        });
      } catch (e) { console.warn("hljs highlight failed:", e); }
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
          checkCompletion();
        }
      };
      window.addEventListener("scroll", onScroll, { passive: true });
      // 记住 listener 以便清理
      ChapterView._scrollListener = onScroll;
    }

    // 章节完成判定：60% 答题正确率 + (滚动到底 80% 或已答完所有题)
    // 答完所有题本身就说明看完了内容（题嵌在文中、答不出来不行）
    //
    // 准确率口径：
    //   - 单选/多选：correct=true/false 直接计入分子分母
    //   - 简答：correct=null（不可自动判分）→ 提交即视为"已参与"，
    //           计入分母当 1、计入分子当 1（参与 = 正确）
    //   这样补完简答题后 60% 门槛仍可触达，不会因为分母小卡死
    //
    // 语言隔离：只统计当前语言的答题（answer.lang === state.language）
    // 中英答题互不干扰；章节 status 是单一字段，任何语言达标后即标记完成
    function checkCompletion() {
      const m = meta();
      const ch = state.progress.chapters[m.id];
      if (!ch || ch.status === "completed") return;
      // 只取当前语言的答题
      const answered = Object.values(ch.quiz || {}).filter(
        (q) => q.lang === state.language
      );
      if (answered.length === 0) return;
      const graded = answered.filter((q) => q.correct === true || q.correct === false);
      const shortAnswered = answered.filter((q) => q.correct === null).length;
      const correct = graded.filter((q) => q.correct === true).length + shortAnswered;
      const total = graded.length + shortAnswered;
      // 没有任何答题记录 → 1（兜底）；否则按上面口径算
      const rate = total === 0 ? 1 : correct / total;
      if (rate < 0.6) return;
      const totalQuizzes = m.quiz_count || 0;
      const allAnswered = totalQuizzes > 0 && answered.length >= totalQuizzes;
      const scrolled = (ch.scroll_percent || 0) >= 80;
      if (!allAnswered && !scrolled) return;
      actions.completeChapter(m.id);
      showToast(t("chapterCompleted") + " 🎉");
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
      if (hljsTimer) clearTimeout(hljsTimer);
    });

    watch(() => props.id, loadChapter);
    watch(() => state.language, loadChapter);
    // 用户答题后立即重判完成状态（不依赖滚动事件）
    watch(
      () => state.progress?.chapters?.[props.id]?.quiz,
      () => checkCompletion(),
      { deep: true }
    );

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
