import CryptoJS from 'crypto-js';
import type { TwoFAAccount } from './2fa';

// 存储键名
const STORAGE_KEY = '2fa-accounts';

// 生成加密密钥（基于用户密码）
function generateKey(password: string): string {
  // 使用密码的SHA-256哈希作为AES密钥
  return CryptoJS.SHA256(password).toString();
}

// 加密数据
export function encryptData(data: any, password: string): string {
  const key = generateKey(password);
  const encrypted = CryptoJS.AES.encrypt(JSON.stringify(data), key).toString();
  return encrypted;
}

// 解密数据
export function decryptData(encryptedData: string, password: string): any {
  try {
    const key = generateKey(password);
    const decrypted = CryptoJS.AES.decrypt(encryptedData, key);
    const plaintext = decrypted.toString(CryptoJS.enc.Utf8);
    return JSON.parse(plaintext);
  } catch (error) {
    throw new Error('密码错误或数据损坏');
  }
}

// 保存2FA账户到LocalStorage
export function saveAccounts(accounts: TwoFAAccount[], password: string): void {
  const encryptedData = encryptData(accounts, password);
  localStorage.setItem(STORAGE_KEY, encryptedData);
}

// 从LocalStorage加载2FA账户
export function loadAccounts(password: string): TwoFAAccount[] {
  const encryptedData = localStorage.getItem(STORAGE_KEY);
  if (!encryptedData) {
    return [];
  }
  return decryptData(encryptedData, password);
}

// 检查是否有存储的账户
export function hasStoredAccounts(): boolean {
  return localStorage.getItem(STORAGE_KEY) !== null;
}
