import { CalculateMetadataFunction, Composition } from 'remotion';
import { HelloWorld } from './HelloWorld';
import { Card1Test } from './Card1Test';
import { Video, type DescJson } from './Video';

// 默认绑定 desc 1.draft.json；用 CLI --props=<json> 可覆盖任意 desc
import descData from '../scripts/desc/1.draft.json';

const defaultDesc = descData as unknown as DescJson;

// 根据 inputProps 动态算 durationInFrames/fps，让 CLI 喂不同 desc 时不卡死
// --props 传入的是整个 JSON 对象（无 desc 包装），因此用 props.duration_frames
// 根据 inputProps 动态算 durationInFrames/fps，让 CLI 喂不同 desc 时不卡死
// 兼容多种 props 来源：
//   1. `pnpm exec remotion render --props X.json`（X = { desc: {...} } 或裸 DescJson）
//   2. `pnpm studio --props X.json`（X = 裸 DescJson，Studio 注入到 window.remotion_inputProps）
//   3. 没传 --props：fallback 到 Composition defaultProps
const calculateMetadata: CalculateMetadataFunction<{ desc: DescJson }> = async ({
  props,
  defaultProps,
}) => {
  // Studio 模式下 --props 注入到 window.remotion_inputProps，
  // calculateMetadata 的 props 参数在 Studio 里可能拿不到，先补一次
  const studioInputProps =
    typeof window !== 'undefined'
      ? (window as any).remotion_inputProps
      : undefined;
  const p: any = props ?? studioInputProps ?? {};
  const dp: any = defaultProps ?? {};
  // 尝试从顶层读 → 从 desc 读 → 从 defaultProps 顶层读 → 从 defaultProps.desc 读
  const fps: number =
    p.fps ?? p.desc?.fps ?? dp.fps ?? dp.desc?.fps ?? 30;
  // 1) 顶层有 duration_frames（裸 desc JSON） → 直接用
  // 2) desc.duration_frames（包装格式） → 直接用
  // 3) 顶层有 cards → 从 cards 重算
  // 4) defaultProps.desc.cards → 从 cards 重算
  let durationInFrames: number | undefined =
    p.duration_frames ?? p.desc?.duration_frames;
  if (durationInFrames == null) {
    const cards = p.cards ?? p.desc?.cards ?? dp.desc?.cards;
    if (Array.isArray(cards)) {
      durationInFrames = cards.reduce(
        (s: number, c: any) => s + Math.round(c.duration_sec * fps),
        0
      );
    }
  }
  return {
    durationInFrames: durationInFrames ?? 1,
    fps,
    width: 1080,
    height: 1920,
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="HelloWorld"
        component={HelloWorld}
        durationInFrames={180}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* Step 8 回归测试：把 desc 1.draft.json 的 card 1 渲染成 6s 视频 */}
      <Composition
        id="Card1Test"
        component={Card1Test}
        durationInFrames={180}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* Step 9/10：端到端视频组合（feed 整份 desc JSON，duration 动态） */}
      {/* durationInFrames/fps 用占位值，最终由 calculateMetadata 从 props 算出 */}
      {/* (Studio 模式 calculateMetadata 可能从 defaultProps 读 fallback) */}
      <Composition
        id="EnSentenceVideo"
        component={Video}
        durationInFrames={1}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{ desc: defaultDesc }}
        calculateMetadata={calculateMetadata}
      />
    </>
  );
};