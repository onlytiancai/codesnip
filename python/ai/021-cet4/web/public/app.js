// 021-cet4 · 前端 Vue 3 应用
// 数据源:GET /api/categories + /api/data + /api/dict/:word
// 语音:POST /api/tts (单例音频 + playingId 高亮)

const { createApp } = Vue

createApp({
  data() {
    return {
      health: null,
      data: null,
      categories: [],
      loadError: null,

      selectedCategoryId: 1, // 默认选中第一个分类
      selectedGroupId: null, // 当前展示的 group_id
      playingId: null,
      audio: null,
      isDesktop: false, // PC 检测(>=1024px);用于区分抽屉行为

      // 抽屉
      drawerOpen: false,
      mobileCategoryDrawerOpen: false,
      dictLoading: false,
      dictError: null,
      dictEntry: null,

      // Toast
      toast: null,
      toastTimer: null,
    }
  },

  computed: {
    totalGroups() {
      return this.data?.total_groups || 0
    },
    totalWords() {
      return this.data?.vocab_size || 0
    },

    /** 当前分类对象（来自 data.categories） */
    currentCategory() {
      if (!this.data) return null
      return this.data.categories.find((c) => c.category_id === this.selectedCategoryId) || null
    },

    /** 当前 group 对象 */
    currentGroup() {
      const cat = this.currentCategory
      if (!cat) return null
      if (this.selectedGroupId == null) return cat.groups[0]
      return cat.groups.find((g) => g.group_id === this.selectedGroupId) || cat.groups[0]
    },

    /** 当前分类下所有 groups（给网格用） */
    siblingGroups() {
      return this.currentCategory?.groups || []
    },

    /** 把 explanation_zh 按 \n\n / \n 切成段 */
    explanationParagraphs() {
      const txt = this.currentGroup?.explanation_zh || ''
      if (!txt) return []
      return txt
        .split(/\n+/)
        .map((s) => s.trim())
        .filter(Boolean)
    },

    /** 抽屉里词条字典 content.word.content */
    dictWordContent() {
      return this.data && this.dictEntry?.content?.word?.content
    },
  },

  methods: {
    groupKey(id) {
      return `group-${id}`
    },
    wordKey(w) {
      return `word-${w}`
    },

    selectCategory(id) {
      this.selectedCategoryId = id
      // 切换分类时自动选该分类第一个 group
      const cat = this.data?.categories.find((c) => c.category_id === id)
      this.selectedGroupId = cat?.groups?.[0]?.group_id ?? null
    },

    selectGroup(id) {
      this.selectedGroupId = id
      // 滚到主面板顶部
      this.$nextTick(() => {
        const el = document.querySelector('[data-main-anchor]')
        if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      })
    },

    async openDrawerForWord(word) {
      // PC:抽屉常驻(layout 第三列),只切 dictEntry
      // mobile:抽屉是 fixed overlay,需要 drawerOpen=true 触发动画
      this.drawerOpen = true
      this.dictLoading = true
      this.dictError = null
      this.dictEntry = null
      try {
        const res = await fetch(`/api/dict/${encodeURIComponent(word)}`, { cache: 'no-store' })
        if (res.status === 404) {
          this.dictError = `未在词典中找到 "${word}"`
          return
        }
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        this.dictEntry = await res.json()
      } catch (e) {
        this.dictError = `查询失败: ${e.message || e}`
      } finally {
        this.dictLoading = false
      }
    },

    closeDrawer() {
      // 桌面端:抽屉是常驻栏,close 表示清空内容让它显示空态
      this.drawerOpen = false
      this.dictEntry = null
      this.dictError = null
      this.dictLoading = false
    },

    showToast(text, ms = 2200) {
      this.toast = text
      clearTimeout(this.toastTimer)
      this.toastTimer = setTimeout(() => {
        this.toast = null
      }, ms)
    },

    async copyWord(w) {
      try {
        await navigator.clipboard.writeText(w)
        this.showToast(`已复制: ${w}`)
      } catch {
        this.showToast('复制失败,请手动选中')
      }
    },

    /** 调用 TTS API 并播放;命中磁盘缓存由后端处理 */
    async playTTS(text, kind, playKey) {
      if (typeof text !== 'string' || !text.trim()) return

      // 单例音频:停止上一段
      if (this.audio) {
        try {
          this.audio.pause()
        } catch {}
        this.audio = null
      }
      this.playingId = playKey

      try {
        const res = await fetch('/api/tts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, kind }),
        })
        if (!res.ok) {
          let err = `HTTP ${res.status}`
          try {
            const data = await res.json()
            err = data.error || err
            if (data.detail) err += ` · ${String(data.detail).slice(0, 200)}`
          } catch {}
          this.showToast(`🔊 TTS 失败: ${err}`, 3500)
          this.playingId = null
          return
        }
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const audio = new Audio(url)
        this.audio = audio
        audio.onended = () => {
          if (this.playingId === playKey) this.playingId = null
          URL.revokeObjectURL(url)
        }
        audio.onerror = () => {
          if (this.playingId === playKey) this.playingId = null
          URL.revokeObjectURL(url)
        }
        await audio.play()
      } catch (e) {
        this.showToast(`🔊 TTS 异常: ${e.message || e}`, 3500)
        this.playingId = null
      }
    },
  },

  watch: {
    selectedCategoryId() {
      // 切换分类时清掉抽屉内容
      this.dictEntry = null
      this.dictError = null
    },
  },

  async mounted() {
    // 检测 PC 视口 (用于抽屉行为分支)
    const mq = window.matchMedia('(min-width: 1024px)')
    this.isDesktop = mq.matches
    mq.addEventListener('change', (e) => {
      this.isDesktop = e.matches
      // 跨临界切换时清掉抽屉
      if (e.matches) {
        this.drawerOpen = false
      } else {
        this.mobileCategoryDrawerOpen = false
      }
    })

    // 健康
    try {
      const h = await fetch('/api/health')
      if (h.ok) this.health = await h.json()
    } catch {
      this.health = { ok: false, hasKey: false }
    }

    // 分类汇总
    try {
      const c = await fetch('/api/categories')
      if (c.ok) this.categories = await c.json()
    } catch {}

    // 完整数据
    try {
      const res = await fetch('/api/data', { cache: 'no-store' })
      if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`)
      this.data = await res.json()
      // 默认选中第一个分类 + 第一个 group
      if (this.data.categories.length) {
        this.selectCategory(this.data.categories[0].category_id)
      }
      document.title = `📚 CET-4 核心词 ${this.data.vocab_size} · 分组浏览`
    } catch (e) {
      this.loadError = e.message || String(e)
    }

    // Esc 关闭抽屉
    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'Escape') {
        if (this.drawerOpen) this.closeDrawer()
        else if (this.mobileCategoryDrawerOpen) this.mobileCategoryDrawerOpen = false
      }
    })
  },
}).mount('#app')
