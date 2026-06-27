import { Composition } from 'remotion';
import { HelloWorld } from './HelloWorld';
import { Card1Test } from './Card1Test';
import { Video, type DescJson } from './Video';

// 默认绑定 desc 1.draft.json；用 CLI --props=<json> 可覆盖（Step 10 写 render 脚本时用）
import descData from '../scripts/desc/1.draft.json';

const defaultDesc = descData as unknown as DescJson;

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
      {/* Step 9：端到端视频组合（喂整份 desc JSON） */}
      <Composition
        id="EnSentenceVideo"
        component={Video}
        durationInFrames={defaultDesc.duration_frames}
        fps={defaultDesc.fps}
        width={1080}
        height={1920}
        defaultProps={{ desc: defaultDesc }}
      />
    </>
  );
};