import { CalculateMetadataFunction, Composition } from 'remotion';
import { HelloWorld } from './HelloWorld';
import { Card1Test } from './Card1Test';
import { Video, type DescJson } from './Video';

// 默认绑定 desc 1.draft.json；用 CLI --props=<json> 可覆盖任意 desc
import descData from '../scripts/desc/1.draft.json';

const defaultDesc = descData as unknown as DescJson;

// 根据 inputProps.desc 动态算 durationInFrames/fps，让 CLI 喂不同 desc 时不卡死
const calculateMetadata: CalculateMetadataFunction<{ desc: DescJson }> = async ({
  props,
}) => ({
  durationInFrames: props.desc.duration_frames,
  fps: props.desc.fps,
  width: 1080,
  height: 1920,
});

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
      <Composition
        id="EnSentenceVideo"
        component={Video}
        durationInFrames={defaultDesc.duration_frames}
        fps={defaultDesc.fps}
        width={1080}
        height={1920}
        defaultProps={{ desc: defaultDesc }}
        calculateMetadata={calculateMetadata}
      />
    </>
  );
};