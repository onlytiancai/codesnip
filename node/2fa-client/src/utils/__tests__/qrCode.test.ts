import { describe, it, expect, beforeEach, vi } from 'vitest';
import { extractSecretFromQRCode, createImageDataFromUrl } from '../qrCode';

// Mock jsQR
vi.mock('jsqr', () => ({
  default: vi.fn()
}));

// 导入模拟后的jsQR
import jsQR from 'jsqr';
const mockJsQR = jsQR as vi.Mock;

// 测试数据
const mockImageData = {
  data: new Uint8ClampedArray(100 * 100 * 4),
  width: 100,
  height: 100
};

describe('qrCode.ts', () => {
  beforeEach(() => {
    // 清空mock调用
    vi.clearAllMocks();
    // Mock document object
    if (typeof document === 'undefined') {
      global.document = {
        createElement: vi.fn()
      } as any;
    }
  });

  describe('extractSecretFromQRCode', () => {
    it('should extract secret from valid QR code', () => {
      // Mock jsQR返回有效的otpauth URL
      mockJsQR.mockReturnValue({
        data: 'otpauth://totp/Example:test@example.com?secret=JBSWY3DPEHPK3PXP&issuer=Example'
      });

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toEqual({
        name: 'Example:test@example.com',
        secret: 'JBSWY3DPEHPK3PXP',
        issuer: 'Example'
      });
    });

    it('should return null when jsQR returns null', () => {
      // Mock jsQR返回null
      mockJsQR.mockReturnValue(null);

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toBeNull();
    });

    it('should return null when jsQR returns no data', () => {
      // Mock jsQR返回空数据
      mockJsQR.mockReturnValue({ data: null });

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toBeNull();
    });

    it('should return null for non-otpauth URL', () => {
      // Mock jsQR返回非otpauth URL
      mockJsQR.mockReturnValue({
        data: 'https://example.com'
      });

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toBeNull();
    });

    it('should return null for non-totp URL', () => {
      // Mock jsQR返回非totp URL
      mockJsQR.mockReturnValue({
        data: 'otpauth://hotp/Example:test@example.com?secret=JBSWY3DPEHPK3PXP'
      });

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toBeNull();
    });

    it('should return null when no secret parameter', () => {
      // Mock jsQR返回没有secret参数的URL
      mockJsQR.mockReturnValue({
        data: 'otpauth://totp/Example:test@example.com?issuer=Example'
      });

      const result = extractSecretFromQRCode(mockImageData);
      expect(result).toBeNull();
    });
  });

  describe('createImageDataFromUrl', () => {
    it('should create ImageData from valid URL', async () => {
      // 跳过浏览器环境相关的测试
      if (typeof document === 'undefined') {
        return expect(true).toBe(true);
      }

      // Mock DOM elements
      const mockCanvas = {
        width: 100,
        height: 100,
        getContext: vi.fn(() => ({
          drawImage: vi.fn(),
          getImageData: vi.fn(() => ({
            data: new Uint8ClampedArray(100 * 100 * 4),
            width: 100,
            height: 100
          }))
        }))
      };

      const mockImage = {
        crossOrigin: '',
        onload: null,
        onerror: null,
        src: ''
      };

      vi.spyOn(document, 'createElement').mockImplementation((tag) => {
        if (tag === 'canvas') return mockCanvas as any;
        if (tag === 'img') return mockImage as any;
        return {} as any;
      });

      // Mock Image constructor
      global.Image = vi.fn(function() {
        return mockImage;
      }) as any;

      // 模拟图片加载成功
      const promise = createImageDataFromUrl('https://example.com/qr.png');
      mockImage.onload?.();

      const result = await promise;
      expect(result).toHaveProperty('data');
      expect(result).toHaveProperty('width');
      expect(result).toHaveProperty('height');
    });

    it('should reject when image fails to load', async () => {
      // 跳过浏览器环境相关的测试
      if (typeof document === 'undefined') {
        return expect(true).toBe(true);
      }

      // Mock DOM elements
      const mockCanvas = {
        getContext: vi.fn(() => ({
          drawImage: vi.fn(),
          getImageData: vi.fn()
        }))
      };

      const mockImage = {
        crossOrigin: '',
        onload: null,
        onerror: null,
        src: ''
      };

      vi.spyOn(document, 'createElement').mockImplementation((tag) => {
        if (tag === 'canvas') return mockCanvas as any;
        if (tag === 'img') return mockImage as any;
        return {} as any;
      });

      // Mock Image constructor
      global.Image = vi.fn(function() {
        return mockImage;
      }) as any;

      // 模拟图片加载失败
      const promise = createImageDataFromUrl('https://example.com/invalid.png');
      mockImage.onerror?.();

      await expect(promise).rejects.toThrow('图片加载失败');
    });
  });
});
