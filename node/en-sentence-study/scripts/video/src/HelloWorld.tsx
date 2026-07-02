import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import { Header } from './components/Header';
import { Footer } from './components/Footer';

type Subtitle = {
  text: string;
  start: number;
  end: number;
  lang: 'en' | 'zh';
};

// 与 TTS 文本逐段对应："Hello, 欢迎使用 Remotion 英语口语视频生成器，环境测试运行成功。"
const SUBS: Subtitle[] = [
  { text: 'Hello,',               start: 0,   end: 21,  lang: 'en' },
  { text: '欢迎使用 Remotion',    start: 21,  end: 66,  lang: 'zh' },
  { text: '英语口语视频生成器',    start: 66,  end: 111, lang: 'zh' },
  { text: '环境测试运行成功',      start: 111, end: 171, lang: 'zh' },
];

const THEME: 'mint' | 'sunny' = 'mint';

export const HelloWorld: React.FC = () => {
  const frame = useCurrentFrame();
  const { fps, durationInFrames } = useVideoConfig();

  // 主标题 spring 缩放 0→1
  const titleScale = spring({ frame, fps, config: { damping: 12 } });
  // 副标题淡入（30~60 帧）
  const subtitleOpacity = interpolate(frame, [30, 60], [0, 1], { extrapolateRight: 'clamp' });
  // 整体淡出（最后 24 帧）
  const exitOpacity = interpolate(
    frame,
    [durationInFrames - 24, durationInFrames],
    [1, 0],
    { extrapolateLeft: 'clamp', extrapolateRight: 'clamp' }
  );
  // 插画淡入（0~30 帧）
  const imageOpacity = interpolate(frame, [0, 30], [0, 0.65], { extrapolateRight: 'clamp' });

  const activeSub = SUBS.filter((s) => frame >= s.start && frame < s.end).slice(-1)[0];

  return (
    <AbsoluteFill style={{ background: '#F4FBF8' }}>
      {/* 底层：AI 插画 */}
      <AbsoluteFill style={{ opacity: imageOpacity }}>
        <Img
          src={staticFile('images/scene.jpg')}
          style={{
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            filter: 'blur(2px)',
          }}
        />
      </AbsoluteFill>

      {/* 中层：极淡的薄荷绿渐变，仅用来统一色调（不挡图） */}
      <AbsoluteFill
        style={{
          background: `linear-gradient(135deg, rgba(230, 247, 240, 0.2) 0%, rgba(244, 251, 248, 0.25) 100%)`,
        }}
      />

      {/* 顶层：主内容 */}
      <AbsoluteFill
        style={{
          opacity: exitOpacity,
          fontFamily: 'system-ui, -apple-system, sans-serif',
        }}
      >
        {/* 主标题 */}
        <div
          style={{
            position: 'absolute',
            top: 280,
            left: 0,
            right: 0,
            transform: `scale(${titleScale})`,
            textAlign: 'center',
          }}
        >
          <h1 style={{ fontSize: 120, color: '#0E3B2E', margin: 0 }}>
            🎬 Hello Remotion
          </h1>
        </div>

        {/* 副标题 */}
        <div
          style={{
            position: 'absolute',
            top: 470,
            left: 0,
            right: 0,
            opacity: subtitleOpacity,
            textAlign: 'center',
          }}
        >
          <p style={{ fontSize: 40, color: '#5A8475', margin: 0 }}>
            英语口语视频生成器 · 环境验证
          </p>
        </div>

        {/* 字幕条 */}
        <div
          style={{
            position: 'absolute',
            bottom: 200,
            left: 60,
            right: 60,
            minHeight: 110,
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            background: 'rgba(14, 59, 46, 0.88)',
            borderRadius: 24,
            padding: '20px 40px',
          }}
        >
          {activeSub && (
            <p
              style={{
                fontSize: activeSub.lang === 'en' ? 72 : 56,
                color: '#FFFFFF',
                margin: 0,
                fontWeight: 600,
                letterSpacing: activeSub.lang === 'en' ? '0.5px' : '4px',
              }}
            >
              {activeSub.text}
            </p>
          )}
        </div>

        {/* TTS 音频 */}
        <Audio src={staticFile('audio/hello.mp3')} />
      </AbsoluteFill>

      {/* 全局 Header / Footer（盖在最上层） */}
      <Header text="英语口语 · 每日一句" theme={THEME} />
      <Footer text="@en-sentence-study" theme={THEME} />
    </AbsoluteFill>
  );
};