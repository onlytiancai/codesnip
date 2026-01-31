// 导入FFmpeg模块（使用动态导入方式）
let FFmpeg: any;
let ffmpegModuleLoaded = false;
let ffmpegModulePromise: Promise<void>;
let ffmpegInstance: any = null;

// 实现fetchFile函数
export const fetchFile = async (blob: Blob): Promise<Uint8Array> => {
  const arrayBuffer = await blob.arrayBuffer();
  return new Uint8Array(arrayBuffer);
};

// 动态导入FFmpeg模块
ffmpegModulePromise = import('@ffmpeg/ffmpeg').then(ffmpegModule => {
  // 使用类型断言来绕过TypeScript错误
  const module = ffmpegModule as any;
  FFmpeg = module.FFmpeg || module.default?.FFmpeg;
  ffmpegModuleLoaded = true;
  console.log('FFmpeg module loaded successfully:', { FFmpeg });
}).catch(error => {
  console.error('Failed to load FFmpeg module:', error);
  ffmpegModuleLoaded = false;
});

// 初始化FFmpeg实例
export const initFFmpeg = async () => {
  // 等待FFmpeg模块加载完成
  if (!ffmpegModuleLoaded) {
    await ffmpegModulePromise;
  }
  
  if (!FFmpeg) {
    throw new Error('Failed to load FFmpeg module');
  }
  
  if (!ffmpegInstance) {
    ffmpegInstance = new FFmpeg({
      log: true,
      progress: (progress: any) => {
        console.log('FFmpeg progress callback:', progress);
      }
    });
  }
  
  return ffmpegInstance;
};

// 合并音频文件
export const mergeAudioFiles = async (audioUrls: string[], wasmFileUrl: string): Promise<Blob | null> => {
  try {
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg
    if (!ffmpeg.loaded) {
      await ffmpeg.load({
        coreURL: import.meta.env.VITE_FFMPEG_CORE_URL ,
        wasmURL: wasmFileUrl
      });
    }
    
    // 为每个音频文件创建临时文件
    for (let i = 0; i < audioUrls.length; i++) {
      const audioUrl = audioUrls[i];
      if (audioUrl) {
        const response = await fetch(audioUrl);
        const blob = await response.blob();
        await ffmpeg.writeFile(`input${i}.wav`, await fetchFile(blob));
      }
    }
    
    // 使用filter_complex方式合并音频，更可靠
    const inputArgs = [];
    const filterArgs = [];
    
    // 构建输入参数和滤镜参数
    for (let i = 0; i < audioUrls.length; i++) {
      inputArgs.push('-i', `input${i}.wav`);
      filterArgs.push(`[${i}:a]`);
    }
    
    // 构建完整的滤镜链
    filterArgs.push('concat=n=' + audioUrls.length + ':v=0:a=1[outa]');
    
    // 使用FFmpeg合并音频
    await ffmpeg.exec([
      ...inputArgs,
      '-filter_complex', filterArgs.join(''),
      '-map', '[outa]',
      'merged_audio.wav'
    ]);
    
    // 读取合并后的音频
    const data = await ffmpeg.readFile('merged_audio.wav');
    const blob = new Blob([data.buffer], { type: 'audio/wav' });
    
    // 清理临时文件
    for (let i = 0; i < audioUrls.length; i++) {
      await ffmpeg.deleteFile(`input${i}.wav`);
    }
    
    return blob;
  } catch (error) {
    console.error('音频合并失败:', error);
    return null;
  }
};

// 生成视频
export const createVideoFromParts = async (imageBlob: Blob, audioBlob: Blob, wasmFileUrl: string): Promise<Blob | null> => {
  try {
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg
    if (!ffmpeg.loaded) {
      await ffmpeg.load({
        coreURL: import.meta.env.VITE_FFMPEG_CORE_URL,
        wasmURL: wasmFileUrl 
      });
    }
    
    // 写入文件到FFmpeg虚拟文件系统
    await ffmpeg.writeFile('image.png', await fetchFile(imageBlob));
    await ffmpeg.writeFile('audio.wav', await fetchFile(audioBlob));
    
    // 使用FFmpeg生成视频
    await ffmpeg.exec([
      '-loop', '1',
      '-i', 'image.png',
      '-i', 'audio.wav',
      '-c:v', 'libx264',
      '-c:a', 'aac',
      '-shortest',
      '-pix_fmt', 'yuv420p',
      '-aspect', '9:16',
      '-progress', 'pipe:3',
      'output.mp4'
    ]);
    
    // 读取生成的视频
    const data = await ffmpeg.readFile('output.mp4');
    const videoBlob = new Blob([data.buffer], { type: 'video/mp4' });
    
    // 清理临时文件
    await ffmpeg.deleteFile('image.png');
    await ffmpeg.deleteFile('audio.wav');
    await ffmpeg.deleteFile('merged_audio.wav');
    await ffmpeg.deleteFile('output.mp4');
    
    return videoBlob;
  } catch (error) {
    console.error('视频合成失败:', error);
    return null;
  }
};
