/* ============================================
   article.js — 文章页 Vue 3 应用
   - 加载 iron-condor-bilingual.md
   - 预处理 ::: en / ::: zh 块 → <div data-lang="..."> 包装
   - marked + DOMPurify 渲染
   - 三种语言模式：双语 / 仅英文 / 仅中文
   - 后处理：双语标题按 " / " 切分为 _zh / _en spans
   ============================================ */
(function () {
  'use strict';

  const { createApp, reactive, onMounted, nextTick } = Vue;

  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });
  }

  // ============================================
  // 预处理 ::: en / ::: zh 块
  // ============================================
  const RE_BLOCK_OPEN = /^:::\s+(en|zh)\s*$/;
  const RE_BLOCK_CLOSE = /^:::\s*$/;

  function preprocessMd(md) {
    const lines = md.split(/\r?\n/);
    const out = [];
    const stack = []; // [{ lang, buf: [] }]
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      const mOpen = line.match(RE_BLOCK_OPEN);
      if (mOpen) {
        stack.push({ lang: mOpen[1], buf: [] });
        i++;
        continue;
      }
      if (RE_BLOCK_CLOSE.test(line) && stack.length) {
        const top = stack.pop();
        // 用空行包住确保 marked 把它当独立块
        out.push(`<div class="lang-block" data-lang="${top.lang}">`);
        out.push('');
        out.push(top.buf.join('\n'));
        out.push('');
        out.push('</div>');
        i++;
        continue;
      }
      if (stack.length) {
        stack[stack.length - 1].buf.push(line);
      } else {
        out.push(line);
      }
      i++;
    }
    // 兜底：未闭合的块 → 当作普通 markdown 输出
    if (stack.length) {
      console.warn('[article] unclosed ::: block(s), treating as neutral:', stack.map(s => s.lang));
      for (const top of stack) {
        out.push(top.buf.join('\n'));
      }
    }
    return out.join('\n');
  }

  function markedSafe(text) {
    if (!text) return '';
    if (typeof marked === 'undefined') {
      const div = document.createElement('div');
      div.textContent = text;
      return div.innerHTML;
    }
    let html;
    try {
      html = marked.parse(String(text));
    } catch (e) {
      console.warn('marked parse error:', e);
      const div = document.createElement('div');
      div.textContent = text;
      html = div.innerHTML;
    }
    if (typeof DOMPurify !== 'undefined') {
      try { return DOMPurify.sanitize(html); } catch (e) { return html; }
    }
    return html;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ============================================
  // 后处理：把双语标题按 " / " 切分
  // ============================================
  function postprocessDom(root) {
    if (!root) return;
    // 标题
    root.querySelectorAll('h1, h2, h3, h4').forEach(h => {
      // 跳过已经在 .lang-block[data-lang] 内部的标题（它们不算双语 split）
      // 实际上 data-lang 内的标题也是单语，不参与 split
      if (h.closest('.lang-block')) return;
      const txt = h.textContent;
      // 优先匹配 " / " 分隔（最常见）
      let sepIdx = txt.indexOf(' / ');
      let sepStr = ' / ';
      let splitKind = 'slash';
      // 退化到 " (xxx)" 括号分隔（用于 H1 文档主标题）
      if (sepIdx === -1) {
        const m = txt.match(/^(.+?)\s*\(([^)]+)\)\s*$/);
        if (m) {
          sepIdx = m[1].length;
          sepStr = ' (';
        } else {
          return;
        }
      }
      const zh = txt.slice(0, sepIdx).trim();
      const en = txt.slice(sepIdx + sepStr.length).replace(/\)\s*$/, '').trim();
      if (!zh || !en) return;
      h.classList.add('bilingual-heading');
      const sepHtml = splitKind === 'slash'
        ? `<span class="_sep"> / </span>`
        : `<span class="_sep"> (</span><span class="_en">)</span>`;
      // 把括号一起塞进 _en：中文模式隐藏 _en 时把括号也带掉
      if (splitKind === 'paren') {
        h.innerHTML =
          `<span class="_zh">${escapeHtml(zh)}</span>` +
          `<span class="_sep"> (</span>` +
          `<span class="_en">${escapeHtml(en)})</span>`;
      } else {
        h.innerHTML =
          `<span class="_zh">${escapeHtml(zh)}</span>` +
          `<span class="_sep"> / </span>` +
          `<span class="_en">${escapeHtml(en)}</span>`;
      }
    });
    // 表格中含 " / " 的单元格
    root.querySelectorAll('table th, table td').forEach(cell => {
      if (cell.closest('.lang-block')) return;
      const txt = cell.textContent;
      if (!txt.includes(' / ')) return;
      const idx = txt.indexOf(' / ');
      const zh = txt.slice(0, idx).trim();
      const en = txt.slice(idx + 3).trim();
      if (!zh || !en) return;
      cell.classList.add('bilingual-cell');
      cell.innerHTML =
        `<span class="_zh">${escapeHtml(zh)}</span>` +
        `<span class="_sep"> / </span>` +
        `<span class="_en">${escapeHtml(en)}</span>`;
    });
    // 译者注 / 免责声明 / 图注 blockquote（单语标记）→ EN/ZH 模式分别隐藏
    root.querySelectorAll('blockquote').forEach(bq => {
      if (bq.closest('.lang-block')) return;
      const txt = bq.textContent;
      if (txt.includes('译者注') || txt.includes('免责声明')) {
        bq.classList.add('zh-only');
      } else if (txt.includes('原文出处') || /^\s*Disclaimer\b/i.test(txt)) {
        bq.classList.add('en-only');
      } else if (/[一-鿿]/.test(txt)) {
        // 含中文字符 → 中文模式显示
        bq.classList.add('zh-only');
      } else {
        // 纯英文/数字 → 英文模式显示
        bq.classList.add('en-only');
      }
    });
    // 目录条目 li 里的 a 链接含 " / "（如 - [引言 / Introduction](#...））
    root.querySelectorAll('li > a').forEach(a => {
      // 跳过已经在 .lang-block[data-lang] 内部的（避免冲突）
      if (a.closest('.lang-block')) return;
      const txt = a.textContent;
      if (!txt.includes(' / ')) return;
      const idx = txt.indexOf(' / ');
      const zh = txt.slice(0, idx).trim();
      const en = txt.slice(idx + 3).trim();
      if (!zh || !en) return;
      const li = a.parentElement;
      li.classList.add('bilingual-link');
      // 只替换 a 内的文本节点，保留 href
      a.innerHTML =
        `<span class="_zh">${escapeHtml(zh)}</span>` +
        `<span class="_sep"> / </span>` +
        `<span class="_en">${escapeHtml(en)}</span>`;
    });
  }

  // ============================================
  // 状态
  // ============================================
  const state = reactive({
    loading: true,
    error: null,
    mode: 'both', // 'both' | 'en' | 'zh'
    rawHtml: '',
    title: '铁鹰式期权完全指南',
  });

  function setMode(m) {
    if (!['both', 'en', 'zh'].includes(m)) return;
    state.mode = m;
    document.documentElement.setAttribute('data-article-mode', m);
  }

  async function loadArticle() {
    try {
      const resp = await fetch('./iron-condor-bilingual.md', { cache: 'no-cache' });
      if (!resp.ok) throw new Error('HTTP ' + resp.status + ' 加载文章失败');
      const md = await resp.text();
      const pre = preprocessMd(md);
      state.rawHtml = markedSafe(pre);
      state.loading = false;
      // 初次设置模式
      setMode('both');
      // 后处理标题切分
      await nextTick();
      const root = document.getElementById('article-root');
      if (root) postprocessDom(root);
    } catch (e) {
      console.error('[article] 加载失败:', e);
      state.error = e.message || String(e);
      state.loading = false;
    }
  }

  // ============================================
  // 组件
  // ============================================
  const ArticleApp = {
    template: `
      <div v-if="state.loading" class="center-screen">
        <div class="spinner"></div>
        <p style="color: var(--text-muted)">正在加载文章…</p>
      </div>
      <div v-else-if="state.error" class="center-screen">
        <div class="error-card">
          <h3>😢 加载失败</h3>
          <p>{{ state.error }}</p>
          <p style="margin-top:12px;font-size:13px">
            请确认你已启动本地 HTTP 服务器（如 <code>pnpm dlx http-server . -p 8765</code>），
            并通过 <code>http://127.0.0.1:8765/article.html</code> 访问。
          </p>
        </div>
      </div>
      <template v-else>
        <header class="article-header">
          <a class="article-back" href="./index.html">📝 答题测试</a>
          <h1 class="article-title">{{ state.title }}</h1>
          <div class="lang-switch" role="tablist" aria-label="语言切换">
            <button type="button" role="tab"
              :class="{ active: state.mode === 'both' }"
              :aria-selected="state.mode === 'both'"
              @click="setMode('both')">中英 / 双语</button>
            <button type="button" role="tab"
              :class="{ active: state.mode === 'en' }"
              :aria-selected="state.mode === 'en'"
              @click="setMode('en')">EN 仅英文</button>
            <button type="button" role="tab"
              :class="{ active: state.mode === 'zh' }"
              :aria-selected="state.mode === 'zh'"
              @click="setMode('zh')">中文 仅中文</button>
          </div>
        </header>
        <article id="article-root" class="article-body" v-html="state.rawHtml"></article>
        <footer class="article-footer">
          <a class="article-back" href="./index.html">← 返回答题测试</a>
        </footer>
      </template>
    `,
    setup() {
      onMounted(loadArticle);
      return { state, setMode };
    },
  };

  // ============================================
  // 挂载
  // ============================================
  const app = createApp({
    template: `<article-app />`,
  });
  app.component('article-app', ArticleApp);
  app.mount('#app');

  // 调试
  window.__article = { state, setMode, loadArticle };
})();
