import { describe, it, expect, beforeEach } from 'vitest';
import { encryptData, decryptData, saveAccounts, loadAccounts, hasStoredAccounts } from '../storage';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString();
    },
    clear: () => {
      store = {};
    },
    removeItem: (key: string) => {
      delete store[key];
    }
  };
})();

// Mock localStorage
globalThis.localStorage = localStorageMock as any;

// 测试数据
const testAccounts = [
  {
    id: '1',
    name: 'test@example.com',
    secret: 'JBSWY3DPEHPK3PXP',
    issuer: 'Example'
  }
];

const testPassword = 'test123';

describe('storage.ts', () => {
  beforeEach(() => {
    // 清空localStorage
    localStorage.clear();
  });

  describe('encryptData and decryptData', () => {
    it('should encrypt and decrypt data correctly', () => {
      const testData = { key: 'value', number: 123, array: [1, 2, 3] };
      const encrypted = encryptData(testData, testPassword);
      const decrypted = decryptData(encrypted, testPassword);
      expect(decrypted).toEqual(testData);
    });

    it('should throw error with wrong password', () => {
      const testData = { key: 'value' };
      const encrypted = encryptData(testData, testPassword);
      expect(() => decryptData(encrypted, 'wrongPassword')).toThrow('密码错误或数据损坏');
    });

    it('should throw error with corrupted data', () => {
      expect(() => decryptData('corruptedData', testPassword)).toThrow('密码错误或数据损坏');
    });
  });

  describe('saveAccounts and loadAccounts', () => {
    it('should save and load accounts correctly', () => {
      saveAccounts(testAccounts, testPassword);
      const loadedAccounts = loadAccounts(testPassword);
      expect(loadedAccounts).toEqual(testAccounts);
    });

    it('should return empty array when no accounts stored', () => {
      const loadedAccounts = loadAccounts(testPassword);
      expect(loadedAccounts).toEqual([]);
    });

    it('should throw error when loading with wrong password', () => {
      saveAccounts(testAccounts, testPassword);
      expect(() => loadAccounts('wrongPassword')).toThrow('密码错误或数据损坏');
    });
  });

  describe('hasStoredAccounts', () => {
    it('should return false when no accounts stored', () => {
      expect(hasStoredAccounts()).toBe(false);
    });

    it('should return true when accounts stored', () => {
      saveAccounts(testAccounts, testPassword);
      expect(hasStoredAccounts()).toBe(true);
    });
  });
});
