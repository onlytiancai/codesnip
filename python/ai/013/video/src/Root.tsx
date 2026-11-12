import { CalculateMetadataFunction, Composition, staticFile } from 'remotion';
import { Video, DescJson } from './Video';
// webpack 会编译期内联 JSON,所以 studio 构建能跑通
import descData from '../../desc.json';

const defaultDesc: DescJson = descData as unknown as DescJson;

const calculateMetadata: CalculateMetadataFunction<{ desc: DescJson }> = async ({
  props,
  defaultProps,
}) => {
  const p: any = props ?? {};
  const dp: any = defaultProps ?? {};
  const descNode = p.desc ?? dp.desc ?? defaultDesc;
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
    width: 1920,
    height: 1080,
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* KaTeX CSS — 必须从 /katex/ 加载,字体路径才能解析正确 */}
      <link rel="stylesheet" href={staticFile('/katex/katex.min.css')} />
      <Composition
        id="XorBPVideo"
        component={Video}
        durationInFrames={defaultDesc.duration_frames > 0
          ? defaultDesc.duration_frames
          : 1}
        fps={defaultDesc.fps}
        width={defaultDesc.width}
        height={defaultDesc.height}
        defaultProps={{ desc: defaultDesc }}
        calculateMetadata={calculateMetadata}
      />
    </>
  );
};
