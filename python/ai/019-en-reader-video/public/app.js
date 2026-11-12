// 019-en-reader-video · 前端 Vue 3 应用
// 数据源:GET /api/explanations  →  语音:POST /api/tts

const { createApp } = Vue;

createApp({
  data() {
    return {
      data: null,
      loadError: null,
      projectName: '',       // ← 来自 /api/project
      currentParagraphIdx: 0,
      currentSentenceIdx: 0,
      playingId: null,
      audio: null,
      apiHealth: null,       // null | 'ok' | 'fail'
    };
  },

  computed: {
    /** 把 preamble + paragraphs 平铺成一个 items 数组(给导航和卡片用) */
    items() {
      if (!this.data) return [];
      const list = [];
      (this.data.preamble || []).forEach((p) => {
        list.push({
          id: p.id,
          type: 'preamble',
          label: p.label || '题注',
          sentences: p.sentences,
          summary: p.summary,
        });
      });
      (this.data.paragraphs || []).forEach((p) => {
        list.push({
          id: p.id,
          type: 'paragraph',
          label: '',
          sentences: p.sentences,
          summary: p.summary,
        });
      });
      return list;
    },

    totalSentences() {
      return this.items.reduce((n, it) => n + (it.sentences?.length || 0), 0);
    },

    /** 上一句按钮是否可点:本段第 0 句之前可以跨段跳到上一段最后一句 */
    canNavigatePrevSentence() {
      if (this.currentSentenceIdx > 0) return true;
      return this.currentParagraphIdx > 0;
    },

    /** 下一句按钮是否可点:本段最后一句之后可以跨段跳到下一段第 0 句 */
    canNavigateNextSentence() {
      const cur = this.items[this.currentParagraphIdx];
      if (cur && this.currentSentenceIdx < cur.sentences.length - 1) return true;
      return this.currentParagraphIdx < this.items.length - 1;
    },

    positionText() {
      const cur = this.items[this.currentParagraphIdx];
      if (!cur) return '';
      const totalPara = this.items.length;
      const totalSent = cur.sentences.length;
      return `第 ${this.currentParagraphIdx + 1}/${totalPara} 段 · S${this.currentSentenceIdx + 1}/${totalSent}`;
    },
  },

  methods: {
    /** 唯一 key,用来驱动 playingId 高亮 */
    sentKey(pi, si, kind) {
      return `${pi}-${si}-${kind}`;
    },

    /** 点击任一句 → 设为当前 (自动展开详情) */
    jumpToSentence(pi, si) {
      this.currentParagraphIdx = pi;
      this.currentSentenceIdx = si;
      this.$nextTick(() => this.scrollActiveIntoView());
    },

    /** 上一段 / 下一段 (jumpSentenceIdx 重置为 0) */
    navigateParagraph(delta) {
      const next = this.currentParagraphIdx + delta;
      if (next < 0 || next >= this.items.length) return;
      this.currentParagraphIdx = next;
      this.currentSentenceIdx = 0;
      this.$nextTick(() => this.scrollActiveIntoView());
    },

    /** 上一句 / 下一句 (可跨段,跳到下一段第 0 句 或 上一段最后一句) */
    navigateSentence(delta) {
      const cur = this.items[this.currentParagraphIdx];
      if (!cur) return;
      const next = this.currentSentenceIdx + delta;

      if (next >= 0 && next < cur.sentences.length) {
        // 段落内移动
        this.currentSentenceIdx = next;
        this.$nextTick(() => this.scrollSentenceIntoView());
        return;
      }
      if (next < 0 && this.currentParagraphIdx > 0) {
        // 跨到上一段最后一句
        this.currentParagraphIdx -= 1;
        const prev = this.items[this.currentParagraphIdx];
        this.currentSentenceIdx = prev.sentences.length - 1;
        this.$nextTick(() => this.scrollActiveIntoView());
        return;
      }
      if (next >= cur.sentences.length && this.currentParagraphIdx < this.items.length - 1) {
        // 跨到下一段第 0 句
        this.currentParagraphIdx += 1;
        this.currentSentenceIdx = 0;
        this.$nextTick(() => this.scrollActiveIntoView());
      }
    },

    scrollActiveIntoView() {
      const el = document.querySelector(`[data-card-idx="${this.currentParagraphIdx}"]`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },

    scrollSentenceIntoView() {
      const el = document.querySelector(`[data-sent-idx="${this.currentParagraphIdx}-${this.currentSentenceIdx}"]`);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' });
    },

    /** 调用 TTS API 并播放;命中磁盘缓存由后端处理 */
    async playTTS(text, kind, playKey) {
      // 停止上一段音频
      if (this.audio) {
        try { this.audio.pause(); } catch {}
        this.audio = null;
      }
      this.playingId = playKey;

      try {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, kind }),
        });
        if (!res.ok) {
          let err = `HTTP ${res.status}`;
          try {
            const data = await res.json();
            err = data.error || err;
            if (data.detail) err += ` · ${data.detail.slice(0, 200)}`;
          } catch {}
          alert(`🔊 TTS 失败: ${err}`);
          this.playingId = null;
          return;
        }
        const blob = await res.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        this.audio = audio;
        audio.onended = () => {
          if (this.playingId === playKey) this.playingId = null;
          URL.revokeObjectURL(url);
        };
        audio.onerror = () => {
          if (this.playingId === playKey) this.playingId = null;
          URL.revokeObjectURL(url);
        };
        await audio.play();
      } catch (e) {
        alert(`🔊 TTS 异常: ${e.message}`);
        this.playingId = null;
      }
    },
  },

  async mounted() {
    try {
      const res = await fetch('/api/explanations', { cache: 'no-store' });
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
      this.data = await res.json();
    } catch (e) {
      this.loadError = e.message || String(e);
    }

    // 检测当前项目名
    try {
      const p = await fetch('/api/project');
      if (p.ok) {
        const info = await p.json();
        this.projectName = info.project || '';
        document.title = `📖 ${info.project} · ${this.data?.metadata?.title_zh || 'en-reader'}`;
      }
    } catch {}

    try {
      const h = await fetch('/api/health');
      this.apiHealth = h.ok ? 'ok' : 'fail';
    } catch {
      this.apiHealth = 'fail';
    }
  },
}).mount('#app');
