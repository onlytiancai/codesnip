import { describe, it, expect } from 'vitest';
import { base32Decode, generateHOTP, generateTOTP, getRemainingTime } from '../2fa';

describe('2fa.ts', () => {
  describe('base32Decode', () => {
    it('should decode base32 string correctly', () => {
      // 测试向量：'JBSWY3DPEHPK3PXP' 应该解码为字节数组
      const result = base32Decode('JBSWY3DPEHPK3PXP');
      expect(result).toBeInstanceOf(Uint8Array);
      // 验证解码结果的长度
      expect(result.length).toBeGreaterThan(0);
    });

    it('should handle empty string', () => {
      const result = base32Decode('');
      expect(result).toBeInstanceOf(Uint8Array);
      expect(result.length).toBe(0);
    });

    it('should handle case insensitivity', () => {
      const result1 = base32Decode('JBSWY3DPEHPK3PXP');
      const result2 = base32Decode('jbswy3dpehpk3pxp');
      expect(result1).toEqual(result2);
    });
  });

  describe('generateHOTP', () => {
    it('should generate HOTP correctly', () => {
      // 测试向量：使用已知的密钥和计数器值
      const secret = 'JBSWY3DPEHPK3PXP';
      const counter = 0;
      const result = generateHOTP(secret, counter);
      expect(result).toMatch(/^\d{6}$/); // 应该是6位数字
    });

    it('should generate different codes for different counters', () => {
      const secret = 'JBSWY3DPEHPK3PXP';
      const code1 = generateHOTP(secret, 0);
      const code2 = generateHOTP(secret, 1);
      expect(code1).not.toEqual(code2);
    });
  });

  describe('generateTOTP', () => {
    it('should generate TOTP correctly', () => {
      const secret = 'JBSWY3DPEHPK3PXP';
      const result = generateTOTP(secret);
      expect(result).toMatch(/^\d{6}$/); // 应该是6位数字
    });

    it('should use custom time step', () => {
      const secret = 'JBSWY3DPEHPK3PXP';
      const result = generateTOTP(secret, 60); // 60秒时间步
      expect(result).toMatch(/^\d{6}$/); // 应该是6位数字
    });
  });

  describe('getRemainingTime', () => {
    it('should return remaining time within time step', () => {
      const timeStep = 30;
      const result = getRemainingTime(timeStep);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(timeStep);
    });

    it('should use custom time step', () => {
      const timeStep = 60;
      const result = getRemainingTime(timeStep);
      expect(result).toBeGreaterThanOrEqual(0);
      expect(result).toBeLessThan(timeStep);
    });
  });
});
