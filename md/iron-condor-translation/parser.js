/* ============================================
   parser.js — 题目 markdown 解析器
   语法：::: quiz <id> <type> ... :::
   ============================================ */
(function (root) {
  'use strict';

  // 正则定义
  const RE_SECTION_COMMENT = /<!--\s*section:\s*([\w\-]+)\s*-->/g;
  const RE_QUIZ_START = /^::: quiz\s+(\S+)\s+(single|multiple)\s*$/;
  const RE_QUIZ_END = /^:::\s*$/;
  const RE_TITLE = /^(#{2,4})\s+(.+?)\s*$/;
  const RE_SIMPLE_KV = /^([a-zA-Z][\w\-]*):\s*(.*)$/;
  const RE_MULTI_KV = /^([a-zA-Z][\w\-]*):\s*\|\s*$/;
  const RE_OPTIONS = /^options:\s*$/;
  // 选项格式：- { key: A, text: "选项文本" }
  // 提取 key 字段值 和 text 字段值（text 在双引号内）
  const RE_OPTION = /^\s*-\s*\{\s*\w+\s*:\s*([\w\-]+)\s*,\s*\w+\s*:\s*"(.+?)"\s*\}\s*$/;

  /**
   * 把任意字符串转成 url/路径安全的 slug
   * 中文保留，标点/空格转 -
   */
  function slugify(s) {
    return (s || '')
      .toString()
      .trim()
      .replace(/\s+/g, '-')
      .replace(/[^\w一-龥\-]+/g, '-')
      .replace(/^-+|-+$/g, '')
      .toLowerCase() || 'section';
  }

  /**
   * 用 marked 解析 + DOMPurify 净化
   */
  function markedSafe(text) {
    if (!text) return '';
    if (typeof marked === 'undefined') {
      // fallback：返回转义后的纯文本
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

  /**
   * 主解析函数
   * @param {string} mdText
   * @returns {{sections: Array, questions: Array}}
   */
  function parseQuiz(mdText) {
    if (!mdText) throw new Error('空文档');
    const lines = mdText.split(/\r?\n/);

    const sections = [];   // [{ key, title, slug, level, questionIds, introLines }]
    const questions = [];  // 完整题目数组

    let currentSection = null;  // 当前标题对象
    let explicitSectionKey = null; // 最近的 <!-- section: key -->
    let q = null;               // 当前累积的题目
    let multiKey = null;        // 多行标量键名
    let multiBuf = [];          // 多行标量缓冲
    let inOptions = false;      // 解析选项中

    function flushMulti() {
      if (q && multiKey) {
        const text = multiBuf.join('\n');
        q[multiKey] = (q[multiKey] || '') + (q[multiKey] ? '\n' : '') + text;
      }
      multiKey = null;
      multiBuf = [];
    }

    function flushQ() {
      if (!q) return;
      flushMulti();
      // 渲染 markdown
      q.questionHtml = markedSafe(q.question);
      q.explanationHtml = markedSafe(q.explanation);
      // 解析 answer
      if (Array.isArray(q.answer)) {
        q.answer = q.answer.join(',');
      }
      q.answer = (q.answer || '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      // 解析 tags
      q.tags = (q.tags || '')
        .split(',')
        .map(s => s.trim())
        .filter(Boolean);
      // 解析 points
      q.points = q.points != null && q.points !== ''
        ? Number(q.points)
        : (q.type === 'multiple' ? 2 : 1);
      // 关联 section
      const sec = sections.find(s => s.key === q.section);
      if (sec) {
        q.sectionTitle = sec.title;
        sec.questionIds.push(q.id);
      } else {
        q.sectionTitle = q.section || '未分类';
      }
      // 选项
      q.options = q.options || [];
      questions.push(q);
      q = null;
      inOptions = false;
    }

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // ----- 标题行（##/###/####） -----
      const mTitle = line.match(RE_TITLE);
      if (mTitle) {
        flushMulti();
        flushQ();
        const level = mTitle[1].length;
        const title = mTitle[2];
        // 用中文部分作为 slug 起点
        const cnPart = title.split('/')[0].trim();
        const slug = slugify(cnPart) || `section-${sections.length}`;
        // 复用同名 section（按 slug 查重）
        let sec = sections.find(s => s.slug === slug);
        if (!sec) {
          sec = {
            key: explicitSectionKey || slug,
            title: title,
            slug: slug,
            level: level,
            questionIds: [],
            introLines: [],
          };
          sections.push(sec);
        } else if (level < sec.level || explicitSectionKey) {
          // 子标题：保留外层 slug，更新 title / level
          sec.title = title;
          sec.level = level;
          if (explicitSectionKey) sec.key = explicitSectionKey;
        }
        currentSection = sec;
        explicitSectionKey = null;
        continue;
      }

      // ----- 章节 key 注释（记录下来，交给下一个标题使用） -----
      const mSection = line.match(/<!--\s*section:\s*([\w\-]+)\s*-->/);
      if (mSection) {
        // 单纯记录，等待下一个标题消费。
        // 注意：不要在这里覆盖 currentSection.key，否则会污染上一个 section。
        explicitSectionKey = mSection[1];
        continue;
      }

      // ----- 题目块开始 -----
      const mQStart = line.match(RE_QUIZ_START);
      if (mQStart) {
        flushMulti();
        flushQ();
        if (!currentSection) {
          // 没有章节上下文：创建一个默认章节
          currentSection = {
            key: 'default',
            title: '未分类',
            slug: 'default',
            level: 2,
            questionIds: [],
            introLines: [],
          };
          sections.push(currentSection);
        }
        q = {
          id: mQStart[1],
          type: mQStart[2],
          section: currentSection.key,
          options: [],
        };
        inOptions = false;
        continue;
      }

      // ----- 题目块结束 -----
      if (RE_QUIZ_END.test(line)) {
        flushQ();
        continue;
      }

      // ----- 不在题目中：跳过 -----
      if (!q) {
        // 收集章节 intro 文本（仅 ## 级别）
        if (currentSection && currentSection.level === 2 && line.trim() !== '') {
          // 简化：只收集非标题非注释的纯文本
          if (!line.match(RE_TITLE) && !line.match(RE_QUIZ_START) && !line.match(RE_SECTION_COMMENT)) {
            currentSection.introLines.push(line);
          }
        }
        continue;
      }

      // ----- 题目字段解析 -----

      // 多行标量开始
      const mMulti = line.match(RE_MULTI_KV);
      if (mMulti) {
        flushMulti();
        multiKey = mMulti[1];
        // 标记：原 key 字段如果已有，保留（多行接续），否则设为空
        if (q[multiKey] == null) q[multiKey] = '';
        inOptions = false;
        continue;
      }

      // options 块开始
      if (RE_OPTIONS.test(line)) {
        flushMulti();
        inOptions = true;
        continue;
      }

      // 单行 KV
      const mSimple = line.match(RE_SIMPLE_KV);
      if (mSimple) {
        const key = mSimple[1];
        const val = mSimple[2];
        // 如果此时正在多行模式（理论上 RE_MULTI_KV 已优先匹配），跳过
        if (multiKey) {
          multiBuf.push(line.replace(/^\s{0,4}/, ''));
          continue;
        }
        if (key === 'options') {
          inOptions = true;
          continue;
        }
        if (inOptions) {
          // 如果之前在 options 模式但又出现 key:value，说明 options 结束
          inOptions = false;
        }
        q[key] = val;
        continue;
      }

      // 选项行（仅在 inOptions 时）
      if (inOptions) {
        const mOpt = line.match(RE_OPTION);
        if (mOpt) {
          q.options.push({ key: mOpt[1], text: mOpt[2] });
          continue;
        }
        // 空行也允许
        if (line.trim() === '') continue;
      }

      // 多行标量内容
      if (multiKey) {
        // 去除至少 2 空格的缩进
        multiBuf.push(line.replace(/^\s{2,}/, ''));
        continue;
      }
    }

    // 收尾
    flushMulti();
    flushQ();

    // 给每章节生成 introHtml
    for (const s of sections) {
      const introMd = s.introLines.join('\n');
      s.introHtml = markedSafe(introMd);
      delete s.introLines; // 不再需要原始行
      // 移除临时字段
      delete s.explicitKey;
    }

    return { sections, questions };
  }

  // 暴露到全局
  root.parseQuiz = parseQuiz;
  // 同时暴露 slugify 供调试
  root.__quizSlugify = slugify;
})(typeof window !== 'undefined' ? window : this);
