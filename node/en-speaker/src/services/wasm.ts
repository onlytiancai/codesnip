import { initFFmpeg } from './ffmpeg';

// 下载wasm文件并显示进度
export const downloadWasm = async (onProgress: (progress: number) => void, onStatus: (status: string) => void): Promise<string> => {
  try {
    onStatus('开始下载wasm文件...');
    onProgress(0);
    
    const wasmUrl = import.meta.env.VITE_FFMPEG_WASM_URL;
    const response = await fetch(wasmUrl, {
      method: 'GET',
      headers: {
        'Accept': '*/*'
      }
    });
    
    if (!response.ok) {
      throw new Error(`下载失败: ${response.statusText}`);
    }
    
    const totalSize = parseInt(response.headers.get('content-length') || '0');
    let downloadedSize = 0;
    const reader = response.body?.getReader();
    
    if (!reader) {
      throw new Error('无法获取响应体');
    }
    
    const chunks: Uint8Array[] = [];
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        break;
      }
      
      chunks.push(value);
      downloadedSize += value.length;
      
      if (totalSize > 0) {
        const progress = Math.round((downloadedSize / totalSize) * 100);
        onProgress(progress);
      }
    }
    
    const blob = new Blob(chunks as BlobPart[], { type: 'application/wasm' });
    const fileUrl = URL.createObjectURL(blob);
    
    onStatus('wasm文件下载成功！');
    onProgress(100);
    
    return fileUrl;
  } catch (error) {
    console.error('下载wasm文件失败:', error);
    const errorMessage = `下载失败: ${error instanceof Error ? error.message : '未知错误'}`;
    onStatus(errorMessage);
    throw error;
  }
};

// 测试wasm是否能正常工作
export const testWasm = async (wasmFileUrl: string, onStatus: (status: string) => void): Promise<boolean> => {
  try {
    onStatus('开始测试wasm...');
    
    // 初始化FFmpeg实例
    const ffmpeg = await initFFmpeg();
    
    // 加载FFmpeg（使用本地下载的wasm文件）
    await ffmpeg.load({
      coreURL: import.meta.env.VITE_FFMPEG_CORE_URL,
      wasmURL: wasmFileUrl
    });
    
    // 简单测试：检查FFmpeg版本
    await ffmpeg.exec(['-version']);
    
    onStatus('wasm测试成功！FFmpeg加载正常。');
    return true;
  } catch (error) {
    console.error('wasm测试失败:', error);
    const errorMessage = `测试失败: ${error instanceof Error ? error.message : '未知错误'}`;
    onStatus(errorMessage);
    return false;
  }
};
