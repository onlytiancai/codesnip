import jsQR from 'jsqr';

// 从二维码中提取2FA密钥
export function extractSecretFromQRCode(imageData: ImageData): { name: string; secret: string; issuer?: string } | null {
  try {
    // 使用jsQR扫描二维码
    const result = jsQR(imageData.data, imageData.width, imageData.height);
    
    if (!result || !result.data) {
      return null;
    }
    
    // 解析二维码数据（otpauth://totp/...格式）
    const url = new URL(result.data);
    
    if (url.protocol !== 'otpauth:' || url.hostname !== 'totp') {
      throw new Error('不是有效的谷歌2FA二维码');
    }
    
    // 提取账户名（从路径中）
    const pathParts = url.pathname.split('/');
    const lastPart = pathParts[pathParts.length - 1];
    const name = decodeURIComponent(lastPart || '');
    
    // 提取密钥
    const secret = url.searchParams.get('secret');
    if (!secret) {
      throw new Error('二维码中未找到密钥');
    }
    
    // 提取 issuer
    const issuer = url.searchParams.get('issuer') || undefined;
    
    return {
      name,
      secret,
      issuer
    };
  } catch (error) {
    console.error('解析二维码失败:', error);
    return null;
  }
}

// 从URL创建ImageData对象（用于测试）
export async function createImageDataFromUrl(url: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx?.drawImage(img, 0, 0);
      const imageData = ctx?.getImageData(0, 0, img.width, img.height);
      if (imageData) {
        resolve(imageData);
      } else {
        reject(new Error('无法创建ImageData'));
      }
    };
    
    img.onerror = () => {
      reject(new Error('图片加载失败'));
    };
    
    img.src = url;
  });
}
