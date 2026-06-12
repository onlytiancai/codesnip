// i18n.js — 简单 i18n 翻译表
// 当前支持：zh / en
// 大段内容用 markdown 文件的双语版本（content/*.md）承载
// 这里只翻译 UI 元素（按钮、提示、章节列表 fallback 等）

export const t = (zh, en, lang) => lang === "en" ? en : zh;

export const I18N = {
  app: {
    title: { zh: "神经网络小课堂", en: "Neural Network 101" },
    tagline: {
      zh: "面试 AI 岗第一题——手写 MLP，从零讲给小学生听",
      en: "Hand-coding an MLP — the #1 interview question, explained for kids",
    },
  },
  ui: {
    home: { zh: "首页", en: "Home" },
    prev: { zh: "← 上一章", en: "← Prev" },
    next: { zh: "下一章 →", en: "Next →" },
    submit: { zh: "提交", en: "Submit" },
    viewAnswer: { zh: "查看答案解析", en: "View Explanation" },
    hideAnswer: { zh: "收起解析", en: "Hide Explanation" },
    reset: { zh: "重做", en: "Reset" },
    answered: { zh: "已答", en: "Done" },
    correct: { zh: "✓ 答对了", en: "✓ Correct" },
    wrong: { zh: "✗ 答错了", en: "✗ Wrong" },
    refAnswer: { zh: "参考答案", en: "Reference Answer" },
    save: { zh: "保存", en: "Save" },
    name: { zh: "你的名字", en: "Your Name" },
    enterName: { zh: "请输入你的名字（用于证书）", en: "Enter your name (for the certificate)" },
    download: { zh: "下载证书 PNG", en: "Download Certificate" },
    print: { zh: "打印证书", en: "Print" },
    lockTitle: { zh: "🔒 证书未解锁", en: "🔒 Certificate Locked" },
    lockDesc: {
      zh: "完成所有 10 章并答对 60% 的题目即可解锁结业证书",
      en: "Complete all 10 chapters and answer 60% of questions correctly",
    },
    locked: { zh: "未解锁", en: "Locked" },
    optional: { zh: "可选", en: "Optional" },
    chaptersTitle: { zh: "章节目录", en: "Chapters" },
    progressTitle: { zh: "学习进度", en: "Progress" },
    minutes: { zh: "分钟", en: "min" },
    correctRate: { zh: "正确率", en: "Accuracy" },
    chapterCompleted: { zh: "已完成", en: "Completed" },
    issuedAt: { zh: "颁发日期", en: "Issued" },
    overallScore: { zh: "总得分", en: "Overall Score" },
    chaptersCompleted: { zh: "完成章节", en: "Chapters" },
    certificateId: { zh: "证书编号", en: "Cert ID" },
    quizLabel: { zh: "测试题", en: "Quiz" },
    quizMultiple: { zh: "多选题", en: "Multiple Choice" },
    quizShort: { zh: "简答题", en: "Short Answer" },
    quizExplain: { zh: "💡 答案解析", en: "💡 Explanation" },
    certTitle: { zh: "结业", en: "Certificate" },
    certLink: { zh: "结业证书", en: "Certificate" },
  },
  quiz: {
    noAnswer: { zh: "请先选择一个选项", en: "Please select an option first" },
    emptyShort: { zh: "请输入你的回答", en: "Please type your answer" },
    submitted: { zh: "已提交", en: "Submitted" },
  },
};

export function pick(dict, lang) {
  return lang === "en" ? dict.en : dict.zh;
}
