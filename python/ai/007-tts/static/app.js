// Kokoro TTS Web Demo - 前端逻辑
(() => {
  'use strict';

  // ---------- 状态 ----------
  const state = {
    voices: [],
    langs: [],
    selectedVoice: 'zf_xiaoni',
    speed: 1.0,
    format: 'wav',
    lastBlobUrl: null,
    lastFileName: null,
    history: [],
  };

  // ---------- DOM ----------
  const $ = (id) => document.getElementById(id);
  const dom = {
    text: $('text-input'),
    charCount: $('char-count'),
    status: $('status'),
    speed: $('speed'),
    speedValue: $('speed-value'),
    formatRadios: document.querySelectorAll('input[name="format"]'),
    btnSynth: $('btn-synth'),
    btnDownload: $('btn-download'),
    btnHistory: $('btn-history'),
    errorBox: $('error-box'),
    voiceGrid: $('voice-grid'),
    voiceSearch: $('voice-search'),
    langFilter: $('lang-filter'),
    player: $('player'),
    playerInfo: $('player-info'),
    historyDrawer: $('history-drawer'),
    historyList: $('history-list'),
    btnCloseHistory: $('btn-close-history'),
  };

  // ---------- 工具 ----------
  const setStatus = (text, cls = 'status-idle') => {
    dom.status.textContent = text;
    dom.status.className = cls;
  };

  const showError = (msg) => {
    if (!msg) {
      dom.errorBox.classList.add('hidden');
      dom.errorBox.textContent = '';
    } else {
      dom.errorBox.classList.remove('hidden');
      dom.errorBox.textContent = msg;
    }
  };

  const formatTime = (ts) => {
    const d = new Date(ts);
    const pad = (n) => String(n).padStart(2, '0');
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
           `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
  };

  // ---------- 加载音色 / 语言 ----------
  const loadVoices = async () => {
    try {
      const [vRes, lRes] = await Promise.all([
        fetch('/api/voices').then((r) => r.json()),
        fetch('/api/langs').then((r) => r.json()),
      ]);
      state.voices = vRes.voices || [];
      state.langs = lRes.langs || [];

      // 填充语言过滤
      dom.langFilter.innerHTML = '<option value="">所有语言</option>' +
        state.langs.map((l) => `<option value="${l.code}">${l.label}</option>`).join('');

      renderVoices();
    } catch (e) {
      console.error(e);
      dom.voiceGrid.innerHTML = `<div class="loading">加载失败：${e.message}</div>`;
    }
  };

  const renderVoices = () => {
    const q = dom.voiceSearch.value.trim().toLowerCase();
    const lang = dom.langFilter.value;
    const filtered = state.voices.filter((v) => {
      if (lang && v.lang_code !== lang) return false;
      if (q) {
        const hay = `${v.name} ${v.label} ${v.lang_label}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }
      return true;
    });

    if (filtered.length === 0) {
      dom.voiceGrid.innerHTML = '<div class="empty">没有匹配的音色</div>';
      return;
    }

    dom.voiceGrid.innerHTML = filtered.map((v) => {
      const isSel = v.name === state.selectedVoice;
      const gender = v.gender === 'F' ? '♀' : v.gender === 'M' ? '♂' : '·';
      return `
        <div class="voice-card ${isSel ? 'selected' : ''}" data-voice="${v.name}">
          <div class="voice-name">${gender} ${v.name}</div>
          <div class="voice-label">${v.label}</div>
          <div class="voice-meta">
            <span class="badge lang">${v.lang_label.split(' ')[0]}</span>
            ${v.quality ? `<span class="badge">Q:${v.quality}</span>` : ''}
          </div>
        </div>
      `;
    }).join('');

    dom.voiceGrid.querySelectorAll('.voice-card').forEach((card) => {
      card.addEventListener('click', () => {
        state.selectedVoice = card.dataset.voice;
        renderVoices();
      });
    });
  };

  // ---------- 合成 ----------
  const synthesize = async () => {
    showError('');
    const text = dom.text.value.trim();
    if (!text) {
      showError('请输入要合成的文本');
      return;
    }
    if (text.length > 2000) {
      showError('文本超过 2000 字符上限');
      return;
    }

    dom.btnSynth.disabled = true;
    dom.btnDownload.disabled = true;
    setStatus('合成中…（首次加载模型会较慢）', 'status-loading');

    const t0 = performance.now();
    try {
      const res = await fetch('/api/tts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          voice: state.selectedVoice,
          speed: state.speed,
          format: state.format,
        }),
      });

      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(detail.detail || `HTTP ${res.status}`);
      }

      const blob = await res.blob();
      const fileName = res.headers.get('X-File-Name') ||
        `tts_${Date.now()}.${state.format}`;

      // 替换播放器源
      if (state.lastBlobUrl) URL.revokeObjectURL(state.lastBlobUrl);
      const url = URL.createObjectURL(blob);
      state.lastBlobUrl = url;
      state.lastFileName = fileName;

      dom.player.src = url;
      dom.player.play().catch(() => { /* 用户未交互时可能自动播放失败 */ });
      const dur = ((performance.now() - t0) / 1000).toFixed(2);
      dom.playerInfo.textContent =
        `${fileName} · ${(blob.size / 1024).toFixed(1)} KB · 合成耗时 ${dur}s`;
      setStatus(`✅ 完成 (${dur}s)`, 'status-ok');

      // 加入历史
      addHistory({
        text, voice: state.selectedVoice, speed: state.speed,
        format: state.format, fileName, size: blob.size, ts: Date.now(),
      });

      // 启用下载
      dom.btnDownload.disabled = false;
    } catch (e) {
      console.error(e);
      showError(`合成失败：${e.message}`);
      setStatus('❌ 失败', 'status-error');
    } finally {
      dom.btnSynth.disabled = false;
    }
  };

  // ---------- 下载 ----------
  const download = () => {
    if (!state.lastBlobUrl || !state.lastFileName) return;
    const a = document.createElement('a');
    a.href = state.lastBlobUrl;
    a.download = state.lastFileName;
    document.body.appendChild(a);
    a.click();
    a.remove();
  };

  // ---------- 历史 ----------
  const addHistory = (item) => {
    state.history.unshift(item);
    if (state.history.length > 50) state.history.pop();
    renderHistory();
    try {
      localStorage.setItem('tts_history', JSON.stringify(state.history));
    } catch {}
  };

  const loadHistory = () => {
    try {
      const raw = localStorage.getItem('tts_history');
      if (raw) state.history = JSON.parse(raw) || [];
    } catch {}
  };

  const renderHistory = () => {
    if (state.history.length === 0) {
      dom.historyList.innerHTML = '<li class="empty">暂无记录</li>';
      return;
    }
    dom.historyList.innerHTML = state.history.map((h, i) => {
      const shortText = h.text.length > 40 ? h.text.slice(0, 40) + '…' : h.text;
      return `
        <li class="history-item" data-idx="${i}">
          <div class="h-text">${escapeHtml(shortText)}</div>
          <div class="h-meta">
            <span class="badge lang">${h.voice}</span>
            <span>·</span>
            <span>${h.format.toUpperCase()}</span>
            <span>·</span>
            <span>${h.speed}×</span>
            <span>·</span>
            <span>${formatTime(h.ts)}</span>
          </div>
        </li>
      `;
    }).join('');

    dom.historyList.querySelectorAll('.history-item').forEach((el) => {
      el.addEventListener('click', () => {
        const h = state.history[+el.dataset.idx];
        if (!h) return;
        dom.text.value = h.text;
        state.selectedVoice = h.voice;
        state.speed = h.speed;
        state.format = h.format;
        dom.speed.value = h.speed;
        dom.speedValue.textContent = `${(+h.speed).toFixed(2)}×`;
        document.querySelectorAll('input[name="format"]').forEach((r) => {
          r.checked = r.value === h.format;
        });
        renderVoices();
        toggleHistory(false);
        updateCharCount();
      });
    });
  };

  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, (c) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));

  const toggleHistory = (force) => {
    const show = force !== undefined ? force : dom.historyDrawer.classList.contains('hidden');
    if (show) {
      dom.historyDrawer.classList.remove('hidden');
      renderHistory();
    } else {
      dom.historyDrawer.classList.add('hidden');
    }
  };

  // ---------- 事件绑定 ----------
  const updateCharCount = () => {
    dom.charCount.textContent = dom.text.value.length;
  };

  const init = () => {
    // 字符计数
    dom.text.addEventListener('input', updateCharCount);
    updateCharCount();

    // 语速
    dom.speed.addEventListener('input', (e) => {
      state.speed = parseFloat(e.target.value);
      dom.speedValue.textContent = `${state.speed.toFixed(2)}×`;
    });

    // 格式
    dom.formatRadios.forEach((r) => {
      r.addEventListener('change', () => {
        if (r.checked) state.format = r.value;
      });
    });

    // 音色搜索 / 语言过滤
    dom.voiceSearch.addEventListener('input', renderVoices);
    dom.langFilter.addEventListener('change', renderVoices);

    // 按钮
    dom.btnSynth.addEventListener('click', synthesize);
    dom.btnDownload.addEventListener('click', download);
    dom.btnHistory.addEventListener('click', () => toggleHistory());
    dom.btnCloseHistory.addEventListener('click', () => toggleHistory(false));

    // 回车 = 合成（Ctrl/⌘+Enter）
    dom.text.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        synthesize();
      }
    });

    // 初始加载
    loadHistory();
    loadVoices();
    setStatus('就绪', 'status-idle');
  };

  document.addEventListener('DOMContentLoaded', init);
})();
