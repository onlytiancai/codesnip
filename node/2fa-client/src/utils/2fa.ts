import jsSHA from 'jssha';

// Base32解码函数
export function base32Decode(base32: string): Uint8Array {
  const base32Chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567';
  let bits = 0;
  let value = 0;
  let index = 0;
  const output = new Uint8Array(Math.ceil(base32.length * 5 / 8));
  
  for (let i = 0; i < base32.length; i++) {
    const char = base32[i];
    if (char) {
      const upperChar = char.toUpperCase();
      const charIndex = base32Chars.indexOf(upperChar);
      
      if (charIndex === -1) continue;
      
      value = (value << 5) | charIndex;
      bits += 5;
      
      if (bits >= 8) {
        bits -= 8;
        output[index++] = (value >>> bits) & 0xFF;
      }
    }
  }
  
  return output.slice(0, index);
}

// 将Uint8Array转换为十六进制字符串
function uint8ArrayToHex(arr: Uint8Array): string {
  return Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('');
}

// HMAC-SHA1计算
export function generateHOTP(secret: string, counter: number): string {
  // 解码Base32密钥
  const decodedSecret = base32Decode(secret);
  
  // 准备计数器值（8字节大端序）
  const buffer = new ArrayBuffer(8);
  const view = new DataView(buffer);
  for (let i = 7; i >= 0; i--) {
    view.setUint8(i, counter & 0xFF);
    counter >>= 8;
  }
  
  // 使用jsSHA计算HMAC-SHA1
  const shaObj = new jsSHA('SHA-1', 'HEX');
  shaObj.setHMACKey(uint8ArrayToHex(decodedSecret), 'HEX');
  shaObj.update(uint8ArrayToHex(new Uint8Array(buffer)));
  const hmac = shaObj.getHMAC('HEX');
  
  // 将十六进制HMAC转换为字节数组
  const hmacBytes = new Uint8Array(hmac.length / 2);
  for (let i = 0; i < hmac.length; i += 2) {
    hmacBytes[i / 2] = parseInt(hmac.substring(i, i + 2), 16);
  }
  
  // 提取动态口令
  const offset = (hmacBytes[hmacBytes.length - 1] as number) & 0x0F;
  const code = (
    (((hmacBytes[offset] as number) & 0x7F) << 24) |
    ((hmacBytes[offset + 1] as number) << 16) |
    ((hmacBytes[offset + 2] as number) << 8) |
    (hmacBytes[offset + 3] as number)
  ) % 1000000;
  
  // 格式化为6位数字
  return code.toString().padStart(6, '0');
}

// 生成TOTP（基于时间的一次性口令）
export function generateTOTP(secret: string, timeStep: number = 30): string {
  // 计算当前时间步
  const currentTime = Math.floor(Date.now() / 1000);
  const counter = Math.floor(currentTime / timeStep);
  
  return generateHOTP(secret, counter);
}

// 获取剩余时间（秒）
export function getRemainingTime(timeStep: number = 30): number {
  const currentTime = Math.floor(Date.now() / 1000);
  return timeStep - (currentTime % timeStep);
}

// 2FA账户接口
export interface TwoFAAccount {
  id: string;
  name: string;
  secret: string;
  issuer?: string;
}
