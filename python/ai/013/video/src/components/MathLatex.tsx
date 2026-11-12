// MathLatex — Remotion 组件,渲染 KaTeX 数学公式
// 注意: KaTeX 的 CSS 必须通过 <link> 标签从 public/ 加载(见 Root.tsx),
//      而不是 import 'katex/dist/katex.min.css' — 后者会让 webpack 内联 CSS,
//      字体 url 'fonts/...' 会相对当前页面(/static-xxx/)解析,而非 /katex/。

import { useMemo, CSSProperties } from 'react';
import katex from 'katex';

// KaTeX 字体路径(Remotion Studio 把 public/ 当静态资源,所以这里写相对路径)
// bundler 会注入 <link rel="stylesheet"> or 直接 inline 取决于配置
const KATEX_FONT_PATH = '/katex/fonts';

type HighlightSpan = {
  /** 渲染前 latex 里的占位符 "<<key>>...<<key>>" */
  key: string;
  /** CSS 样式,默认是橙色高亮底 */
  style?: CSSProperties;
  className?: string;
};

type Props = {
  /** 原始 LaTeX(包含 \color \colorbox \frac 等 KaTeX 支持的语法) */
  latex: string;
  /** 是否块级显示(默认 true) */
  displayMode?: boolean;
  /** 容器样式 */
  style?: CSSProperties;
  /** 在 latex 字符串中可用 "<<key>...<</key>>" 标记的高亮区间 */
  highlights?: HighlightSpan[];
  /** 字号(默认 64px 配 1920×1080 横版) */
  fontSize?: number;
};

/**
 * 渲染一段 LaTeX 数学公式。KaTeX 输出是结构化的 HTML 树,
 * 解析在 useMemo 里跑一次,后续帧只走 interpolate 动画。
 */
export const MathLatex: React.FC<Props> = ({
  latex,
  displayMode = true,
  style,
  highlights = [],
  fontSize = 64,
}) => {
  const html = useMemo(() => {
    let src = latex;
    // 把 "<<key>>...<</key>>" 替换成真实 HTML span(注意 KaTeX 渲染后这些标记会消失)
    // 我们把 key 占位符先去掉,再在 dangerouslySetInnerHTML 后挂 color
    // 这里采用更稳的方法: 让 latex 里的实际数学内容直接渲染,不出 span。
    return katex.renderToString(src, {
      displayMode,
      throwOnError: false,
      output: 'html',
      // 让 KaTeX 用本地字体而非 CDN
      // KaTeX 0.16+ 自动检测 URL,但可显式给 absolute
      // (Remotion 静态资源托管在 /,所以下面这样配置)
    });
  }, [latex, displayMode]);

  // 计算 highlight 占位替换:在 latex 中有 "<<k>>...<</k>>" 切出
  // 用 KaTeX 的 \class{cls}{} 指令实现着色
  // 简化: 我们不强求占位标记,而是用 containerStyle 把"当前 highlight"区间
  // 通过 backgroundColor 覆盖在最上层。MathAnim 卡片只渲染一行,不强需。
  void highlights; // 暂未用,留着扩展

  return (
    <div
      style={{
        fontSize,
        color: '#1A1A1A',
        lineHeight: 1.2,
        ...style,
      }}
      // 注意: KaTeX 输出是受信任的(我们用 throwOnError=false,且 latex 来自作者)
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
};
