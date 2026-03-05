# 程序员英语阅读应用

一个专为程序员设计的英语阅读学习应用，提供沉浸式英文阅读体验和逐句跟读练习。

## 核心功能

- 📚 沉浸式英文阅读体验
- 🎯 逐句跟读练习，AI 语音反馈
- 📖 单词即点即查
- 🔗 文章分享功能

## 技术栈

| 组件 | 技术 |
|------|------|
| 前端框架 | Nuxt 4 |
| UI 框架 | Nuxt UI v4 + Tailwind 4 |
| ORM | Prisma |
| 数据库 | SQLite (开发) / MySQL (生产) |
| 小程序 | 微信小程序 |
| 测试 | Vitest |

## 项目结构

```
wawakeji-wx-app/
├── apps/
│   ├── web/               # Nuxt 4 Web 应用
│   ├── admin/             # 管理员 Dashboard
│   └── miniapp/           # 微信小程序
├── packages/
│   ├── database/          # Prisma Schema
│   ├── shared/            # 共享类型和工具
│   └── testing/           # 测试配置
└── docs/                  # 文档与计划
```

## 开发阶段

- [x] Phase 1: MVP 基础架构
- [ ] Phase 2: 核心跟读功能
- [ ] Phase 3: 认证与管理后台
- [ ] Phase 4: 小程序端
- [ ] Phase 5: 付费与分享
- [ ] Phase 6: 优化与部署

## 快速开始

```bash
# 安装依赖
pnpm install

# 开发 Web 应用
pnpm dev:web

# 运行测试
pnpm test
```

## License

MIT
