## 纯浏览器端的 2FA 客户端

纯浏览器端实现 TOTP 2FA 认证, 不向服务器发送任何信息。

- 动态口令种子加密存储在浏览器本地。
- 所有账户可以加密导出，不会存储在第三方，自主备份和同步。
- 所有加密算法使用浏览器原生的 Web Crypto API，保证安全。

### 开发

技术栈

    typescript + pnpm + vite + vue3 + tailwind4

开发

    pnpm dev:all      

部署

    vi .env
    VITE_BASE_PATH=/2fa/

    pnpm build
    rsync -avP dist/ ihuhao:/home/ubuntu/src/html/2fa

