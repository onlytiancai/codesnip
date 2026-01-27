import type { TwoFAAccount } from './2fa';
import { saveAccounts, loadAccounts } from './storage';

// 添加新账户
export function addAccount(accounts: TwoFAAccount[], name: string, secret: string, issuer?: string): TwoFAAccount[] {
  const newAccount: TwoFAAccount = {
    id: crypto.randomUUID(),
    name,
    secret,
    issuer
  };
  return [...accounts, newAccount];
}

// 删除账户
export function removeAccount(accounts: TwoFAAccount[], accountId: string): TwoFAAccount[] {
  return accounts.filter(account => account.id !== accountId);
}

// 更新账户
export function updateAccount(accounts: TwoFAAccount[], accountId: string, updates: Partial<TwoFAAccount>): TwoFAAccount[] {
  return accounts.map(account => {
    if (account.id === accountId) {
      return { ...account, ...updates };
    }
    return account;
  });
}

// 保存账户到存储
export function saveAccountList(accounts: TwoFAAccount[], password: string): void {
  saveAccounts(accounts, password);
}

// 从存储加载账户
export function loadAccountList(password: string): TwoFAAccount[] {
  return loadAccounts(password);
}
