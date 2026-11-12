// RichText — 行内数学+纯文本混排
// 约定: 字符串里的 [[latex:<inline_latex>]] 当作行内数学,其余当 React 文本(whiteSpace:pre-wrap 保留 \n)
// 渲染: text 段用 <span>;math 段用 <MathLatex displayMode={false}>
// 注意:
//   - KaTeX 解析失败由 MathLatex 的 throwOnError:false 兜底,不会整段崩
//   - 反斜杠语法遵循 desc.json card[8] 既有约定 — latex 反斜杠要写两遍 \\dfrac 等

import { useMemo, CSSProperties } from 'react';
import { MathLatex } from './MathLatex';

const LATEX_PATTERN = /\[\[latex:([\s\S]+?)\]\]/g;

export type RichTextSegment =
  | { kind: 'text'; value: string }
  | { kind: 'inline_math'; value: string };

export type RichTextInput = string | RichTextSegment[];

const parseToSegments = (input: RichTextInput): RichTextSegment[] => {
  if (typeof input !== 'string') return input;
  const out: RichTextSegment[] = [];
  let lastIndex = 0;
  // 重置 lastIndex
  LATEX_PATTERN.lastIndex = 0;
  let m: RegExpExecArray | null;
  while ((m = LATEX_PATTERN.exec(input)) !== null) {
    if (m.index > lastIndex) {
      out.push({ kind: 'text', value: input.slice(lastIndex, m.index) });
    }
    out.push({ kind: 'inline_math', value: m[1] });
    lastIndex = m.index + m[0].length;
  }
  if (lastIndex < input.length) {
    out.push({ kind: 'text', value: input.slice(lastIndex) });
  }
  return out;
};

type Props = {
  input: RichTextInput;
  fontSize?: number;
  /** 行内 KaTeX 字号 = fontSize * inlineScale,默认 0.7(行间数学约 70% 行高) */
  inlineScale?: number;
  color?: string;
  lineHeight?: number;
  fontFamily?: string;
  style?: CSSProperties;
  fontWeight?: CSSProperties['fontWeight'];
  align?: CSSProperties['textAlign'];
};

export const RichText: React.FC<Props> = ({
  input,
  fontSize = 32,
  inlineScale = 0.7,
  color,
  lineHeight = 1.45,
  fontFamily,
  style,
  fontWeight,
  align,
}) => {
  const segments = useMemo(() => parseToSegments(input), [input]);

  return (
    <span
      style={{
        whiteSpace: 'pre-wrap',
        color,
        lineHeight,
        fontFamily,
        fontWeight,
        textAlign: align,
        fontSize,
        ...style,
      }}
    >
      {segments.map((seg, i) => {
        if (seg.kind === 'text') {
          return <span key={i}>{seg.value}</span>;
        }
        return (
          <span
            key={i}
            style={{
              display: 'inline-block',
              verticalAlign: 'middle',
              margin: '0 4px',
              fontSize: fontSize * inlineScale,
              lineHeight: 1,
            }}
          >
            <MathLatex
              latex={seg.value}
              displayMode={false}
              fontSize={fontSize * inlineScale}
              style={{ color: color ?? '#1A1A1A' }}
            />
          </span>
        );
      })}
    </span>
  );
};
