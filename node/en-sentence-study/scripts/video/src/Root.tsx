import { CalculateMetadataFunction, Composition } from 'remotion';
import { HelloWorld } from './HelloWorld';
import { Card1Test } from './Card1Test';
import { Video, type DescJson } from './Video';

import descData from '../scripts/desc/13.draft.json';

const defaultDesc = descData as unknown as DescJson;

const calculateMetadata: CalculateMetadataFunction<{ desc: DescJson }> = async ({
  props,
  defaultProps,
}) => {
  const p: any = props ?? {};
  const dp: any = defaultProps ?? {};
  // 只从 desc 包装里取；不要从 props 顶层 / window / Studio 黑魔法拿
  const descNode = p.desc ?? dp.desc;
  const fps: number = descNode?.fps ?? 30;
  let durationInFrames: number | undefined = descNode?.duration_frames;
  if (durationInFrames == null && Array.isArray(descNode?.cards)) {
    durationInFrames = descNode.cards.reduce(
      (s: number, c: any) => s + Math.round(c.duration_sec * fps),
      0
    );
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
      {/* Step 8 回归测试 */}
      <Composition
        id="Card1Test"
        component={Card1Test}
        durationInFrames={180}
        fps={30}
        width={1080}
        height={1920}
      />
      {/* Step 9/10：端到端视频组合（feed 整份 desc JSON，duration 动态） */}
      {/* durationInFrames/fps 用占位值，最终由 calculateMetadata 从 props.desc 算出 */}
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