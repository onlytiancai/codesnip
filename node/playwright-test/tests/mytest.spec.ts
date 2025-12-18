import { test, expect } from '@playwright/test';

test('test', async ({ page }) => {
  await page.goto('https://www.baidu.com/');
  await page.locator('#head_wrapper').click();
  await page.getByRole('link', { name: '登录' }).click();
  await page.getByText('用户名登录').click();
  await page.getByRole('textbox', { name: '手机号/用户名/邮箱' }).click();
  await page.getByRole('textbox', { name: '手机号/用户名/邮箱' }).fill('aaa');
  await page.getByRole('textbox', { name: '密码' }).click();
  await page.getByRole('textbox', { name: '密码' }).fill('123');
  await page.getByRole('button', { name: '登录' }).click();
});
