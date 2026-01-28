import { describe, it, expect, beforeEach, vi } from 'vitest';
import { addAccount, removeAccount, updateAccount, saveAccountList, loadAccountList } from '../accountManager';
import { saveAccounts, loadAccounts } from '../storage';

// Mock storage functions
vi.mock('../storage', () => ({
  saveAccounts: vi.fn(),
  loadAccounts: vi.fn()
}));

// Mock crypto.randomUUID
if (typeof crypto === 'undefined' || !crypto.randomUUID) {
  Object.defineProperty(global, 'crypto', {
    value: {
      randomUUID: vi.fn(() => 'mocked-uuid')
    },
    writable: true
  });
} else {
  vi.spyOn(crypto, 'randomUUID').mockReturnValue('mocked-uuid');
}

// 测试数据
const initialAccounts = [
  {
    id: '1',
    name: 'test1@example.com',
    secret: 'JBSWY3DPEHPK3PXP',
    issuer: 'Example1'
  }
];

const testPassword = 'test123';

describe('accountManager.ts', () => {
  beforeEach(() => {
    // 清空mock调用
    vi.clearAllMocks();
  });

  describe('addAccount', () => {
    it('should add a new account with generated id', () => {
      const newAccounts = addAccount(initialAccounts, 'test2@example.com', 'ABCDEFGHIJKLMNOP', 'Example2');
      expect(newAccounts).toHaveLength(2);
      expect(newAccounts[1]).toEqual({
        id: 'mocked-uuid',
        name: 'test2@example.com',
        secret: 'ABCDEFGHIJKLMNOP',
        issuer: 'Example2'
      });
    });

    it('should add a new account without issuer', () => {
      const newAccounts = addAccount(initialAccounts, 'test2@example.com', 'ABCDEFGHIJKLMNOP');
      expect(newAccounts).toHaveLength(2);
      expect(newAccounts[1]).toEqual({
        id: 'mocked-uuid',
        name: 'test2@example.com',
        secret: 'ABCDEFGHIJKLMNOP'
        // issuer should be undefined
      });
      expect(newAccounts[1].issuer).toBeUndefined();
    });
  });

  describe('removeAccount', () => {
    it('should remove account by id', () => {
      const newAccounts = removeAccount(initialAccounts, '1');
      expect(newAccounts).toHaveLength(0);
    });

    it('should not change accounts when id not found', () => {
      const newAccounts = removeAccount(initialAccounts, 'non-existent-id');
      expect(newAccounts).toEqual(initialAccounts);
    });
  });

  describe('updateAccount', () => {
    it('should update account by id', () => {
      const updates = {
        name: 'updated@example.com',
        issuer: 'UpdatedIssuer'
      };
      const updatedAccounts = updateAccount(initialAccounts, '1', updates);
      expect(updatedAccounts[0]).toEqual({
        ...initialAccounts[0],
        ...updates
      });
    });

    it('should not change accounts when id not found', () => {
      const updates = {
        name: 'updated@example.com'
      };
      const updatedAccounts = updateAccount(initialAccounts, 'non-existent-id', updates);
      expect(updatedAccounts).toEqual(initialAccounts);
    });
  });

  describe('saveAccountList and loadAccountList', () => {
    it('should call saveAccounts with correct parameters', () => {
      saveAccountList(initialAccounts, testPassword);
      expect(saveAccounts).toHaveBeenCalledWith(initialAccounts, testPassword);
    });

    it('should call loadAccounts and return its result', () => {
      const mockResult = [{ id: '2', name: 'loaded@example.com', secret: 'SECRET' }];
      (loadAccounts as vi.Mock).mockReturnValue(mockResult);
      
      const result = loadAccountList(testPassword);
      expect(loadAccounts).toHaveBeenCalledWith(testPassword);
      expect(result).toEqual(mockResult);
    });
  });
});
