// 进度判定阈值与可调参数
window.PROGRESS_CONFIG = {
  // 章节完成判定：答对 60% 题目 + 滚动到底部
  CHAPTER_COMPLETE_THRESHOLD: 0.6,
  CHAPTER_MIN_SCROLL_PERCENT: 80,

  // 证书颁发门槛
  CERT_PASS_RATE: 0.6,
  CERT_REQUIRE_ALL_CHAPTERS: true,

  // 进度展示
  SHOW_TIME_ESTIMATED: true,

  // localStorage 键名
  STORAGE_KEYS: {
    THEME: '009-theme',
    LANG: '009-lang',
    PROGRESS: '009-progress',
    CERT: '009-cert',
    STUDENT_NAME: '009-student-name',
  },

  // 进度 schema 版本（未来加字段时升级以触发数据迁移）
  // v2: 章节状态按语言拆分（ch.zh / ch.en），quiz key 加 :lang 后缀
  PROGRESS_SCHEMA_VERSION: 2,
};
