import { ExpressionCard } from './components/cards/ExpressionCard';

// Step 8 临时回归测试：硬编码 desc 1.draft.json 的 card 1 数据
// 用 ExpressionCard 渲染 6s 视频，验证布局/音频/动效是否正常
// Step 9 实现 Video.tsx 后会删掉这个文件
const CARD1_DATA = {
  index: 1,
  type: 'expression' as const,
  style: 'polite' as const,
  literal_translation: '您能把这个礼品包装一下吗？',
  sentence_en: 'Could you gift wrap this, please?',
  phonetic: 'kuːd juː ɡɪft ræp ðɪs pliːz',
  note: '最常用且礼貌的说法，gift wrap 是固定搭配',
  tts_segments: [
    {
      lang: 'zh' as const,
      text: '您能把这个礼品包装一下吗？',
      audio_path: '/Users/huhao/src/codesnip/node/en-sentence-study/scripts/video/public/audio/1/1-0-zh.mp3',
      duration_ms: 2334,
    },
    {
      lang: 'en' as const,
      text: 'Could you gift wrap this, please?',
      audio_path: '/Users/huhao/src/codesnip/node/en-sentence-study/scripts/video/public/audio/1/1-1-en.mp3',
      duration_ms: 2200,
    },
  ],
};

export const Card1Test: React.FC = () => {
  return <ExpressionCard card={CARD1_DATA} theme="mint" />;
};