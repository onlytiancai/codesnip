# 英语口语学习 Web App - MVP 实现计划

## Context

用户想要创建一个英语口语学习 web app，核心目标是帮助用户锻炼口语表达能力——心里想说的话能够用英文表达出来。当前是空目录，需要从零开始构建 MVP 版本。

MVP 选择理由：功能范围大，先做核心功能验证产品方向，再逐步完善。

## 技术栈

- **框架**: Nuxt v4
- **UI**: Nuxt UI
- **数据库**: SQLite (better-sqlite3)
- **AI**: OpenAI GPT-4o mini
- **样式**: Tailwind CSS (Nuxt UI 内置)
- **适配**: PC + 移动端自适应

---

## 一、项目初始化

### 1.1 初始化 Nuxt4 项目

```bash
cd /Users/huhao/src/codesnip/node/en-sentence-study
npx nuxi@latest init . --force
```

**关键依赖**:
```json
{
  "dependencies": {
    "@nuxt/ui": "^4.5.0",
    "@nuxt/icon": "^2.2.1",
    "nuxt": "^4.3.1",
    "nuxt-auth-utils": "^0.5.29",
    "better-sqlite3": "^12.6.2",
    "@prisma/client": "^6.19.2",
    "@prisma/adapter-better-sqlite3": "^6.19.2",
    "openai": "^4.0.0",
    "zod": "^4.3.6",
    "bcryptjs": "^3.0.3"
  },
  "devDependencies": {
    "@iconify-json/lucide": "^1.2.94",
    "@types/better-sqlite3": "^7.6.13",
    "@types/bcryptjs": "^3.0.0",
    "prisma": "^6.19.2"
  }
}
```

### 1.2 关键配置文件

**`nuxt.config.ts`**:
```typescript
export default defineNuxtConfig({
  compatibilityDate: "2025-07-15",
  devtools: { enabled: true },
  modules: [
    ["@nuxt/ui", { fonts: false }],
    "@nuxt/icon",
    "nuxt-auth-utils"
  ],
  css: ["~/assets/css/main.css"],
  runtimeConfig: {
    openaiApiKey: process.env.OPENAI_API_KEY,
    oauth: {
      google: {
        clientId: process.env.GOOGLE_CLIENT_ID,
        clientSecret: process.env.GOOGLE_CLIENT_SECRET,
      },
      wechat: {
        clientId: process.env.WECHAT_CLIENT_ID,
        clientSecret: process.env.WECHAT_CLIENT_SECRET,
      },
      wechatMp: {
        appid: process.env.WECHAT_MP_APPID,
        secret: process.env.WECHAT_MP_SECRET,
      }
    }
  }
})
```

**`prisma/schema.prisma`**:
```prisma
datasource db {
  provider = "sqlite"
  url      = "file:./dev.db"
}

generator client {
  provider = "prisma-client-js"
}

model User {
  id        Int      @id @default(autoincrement())
  email     String?  @unique
  password  String?
  nickname  String?
  avatar    String?
  userCode  String   @unique
  googleId  String?  @unique
  wxOpenid  String?  @unique
  createdAt DateTime @default(now())
  updatedAt DateTime @updatedAt
  progress  UserProgress[]
  records   PracticeRecord[]
  reviewSettings ReviewSettings?
  learnedSentences LearnedSentence[]
}

model Scene {
  id          Int     @id @default(autoincrement())
  slug        String  @unique
  nameEn      String
  nameZh      String
  description String?
  icon        String?
  sortOrder   Int     @default(0)
  tasks       Task[]
}

model Task {
  id        Int       @id @default(autoincrement())
  sceneId   Int
  slug      String    @unique
  nameEn    String
  nameZh    String
  sortOrder Int       @default(0)
  scene     Scene     @relation(fields: [sceneId], references: [id])
  sentences Sentence[]
}

model Sentence {
  id        Int      @id @default(autoincrement())
  taskId    Int
  sentenceZh String
  sentenceEn String
  keyWords   String   // JSON
  difficulty Int      @default(1)
  isActive   Boolean  @default(true)
  task       Task     @relation(fields: [taskId], references: [id])
  progress   UserProgress[]
  records    PracticeRecord[]
  learned    LearnedSentence[]
}

model UserProgress {
  id            Int    @id @default(autoincrement())
  userId        Int
  sentenceId    Int
  status        String @default("learning")
  timesPracticed Int   @default(0)
  timesCorrect   Int   @default(0)
  lastPracticedAt DateTime?
  user          User     @relation(fields: [userId], references: [id])
  sentence      Sentence @relation(fields: [sentenceId], references: [id])
  @@unique([userId, sentenceId])
}

model PracticeRecord {
  id        Int      @id @default(autoincrement())
  userId    Int
  sentenceId Int
  userInput String
  aiFeedback String?
  score      Int?
  createdAt  DateTime @default(now())
  user       User     @relation(fields: [userId], references: [id])
  sentence   Sentence @relation(fields: [sentenceId], references: [id])
}

model ReviewSettings {
  id           Int     @id @default(autoincrement())
  userId       Int     @unique
  mode         String  @default("both")
  autoPlay     Boolean @default(true)
  skipLearned  Boolean @default(true)
  user         User    @relation(fields: [userId], references: [id])
}

model LearnedSentence {
  id         Int      @id @default(autoincrement())
  userId     Int
  sentenceId Int
  learnedAt  DateTime @default(now())
  user       User     @relation(fields: [userId], references: [id])
  sentence   Sentence @relation(fields: [sentenceId], references: [id])
  @@unique([userId, sentenceId])
}
```

---

## 二、数据库设计

使用 `server/database/` 目录存放数据库相关代码。

### 2.1 核心表结构

```sql
-- 用户表
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE,
  password_hash TEXT,
  nickname TEXT,
  avatar_url TEXT,
  user_code TEXT UNIQUE,
  google_id TEXT UNIQUE,
  wx_openid TEXT UNIQUE,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 场景表
CREATE TABLE scenes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  slug TEXT UNIQUE NOT NULL,
  name_en TEXT NOT NULL,
  name_zh TEXT NOT NULL,
  description TEXT,
  icon TEXT,
  sort_order INTEGER DEFAULT 0
);

-- 任务表
CREATE TABLE tasks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  scene_id INTEGER NOT NULL,
  slug TEXT UNIQUE NOT NULL,
  name_en TEXT NOT NULL,
  name_zh TEXT NOT NULL,
  sort_order INTEGER DEFAULT 0,
  FOREIGN KEY (scene_id) REFERENCES scenes(id)
);

-- 句子表
CREATE TABLE sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  task_id INTEGER NOT NULL,
  sentence_zh TEXT NOT NULL,
  sentence_en TEXT NOT NULL,
  key_words TEXT NOT NULL,
  difficulty INTEGER DEFAULT 1,
  is_active BOOLEAN DEFAULT 1,
  FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- 用户学习进度
CREATE TABLE user_progress (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  sentence_id INTEGER NOT NULL,
  status TEXT DEFAULT 'learning',
  times_practiced INTEGER DEFAULT 0,
  times_correct INTEGER DEFAULT 0,
  last_practiced_at DATETIME,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (sentence_id) REFERENCES sentences(id),
  UNIQUE(user_id, sentence_id)
);

-- 练习记录
CREATE TABLE practice_records (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  sentence_id INTEGER NOT NULL,
  user_input TEXT NOT NULL,
  ai_feedback TEXT,
  score INTEGER,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 复习设置
CREATE TABLE review_settings (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  mode TEXT DEFAULT 'both',
  auto_play BOOLEAN DEFAULT 1,
  skip_learned BOOLEAN DEFAULT 1,
  FOREIGN KEY (user_id) REFERENCES users(id)
);

-- 学会的句子
CREATE TABLE learned_sentences (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  sentence_id INTEGER NOT NULL,
  learned_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (sentence_id) REFERENCES sentences(id),
  UNIQUE(user_id, sentence_id)
);
```

---

## 三、页面结构

```
pages/
├── index.vue                    # 首页仪表盘
├── login.vue                    # 登录
├── register.vue                 # 注册
├── scenes/
│   ├── index.vue                # 场景列表
│   └── [slug]/
│       ├── index.vue            # 场景详情（任务列表）
│       └── [taskSlug].vue       # 任务练习页
├── practice/
│   └── index.vue                # 随机练习
├── review/
│   └── index.vue                # 闪卡复习
├── profile/
│   └── index.vue                # 个人中心
└── stats/
    └── index.vue                # 学习统计

layouts/
├── default.vue                  # 默认布局
└── auth.vue                     # 认证布局
```

---

## 四、核心 API

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/auth/register` | 邮箱注册 |
| POST | `/api/auth/login` | 邮箱登录 |
| GET | `/api/auth/me` | 获取当前用户 |
| GET | `/api/scenes` | 获取所有场景 |
| GET | `/api/scenes/:slug` | 获取场景详情 |
| GET | `/api/practice/random` | 随机获取句子 |
| POST | `/api/practice/submit` | 提交练习，AI 点评 |
| GET | `/api/hints/:sentenceId` | 获取候选句子 |
| GET | `/api/review/cards` | 获取复习卡片 |
| POST | `/api/review/learn/:id` | 标记学会 |
| GET | `/api/stats/daily` | 本日统计 |
| GET | `/api/stats/weekly` | 本周统计 |

---

## 五、AI Prompt 设计

### 5.1 AI 点评

```
你是一位专业的英语口语教练。点评用户输入的英文句子。

场景：{{sceneName}} ({{sceneNameEn}})
任务：{{taskName}}
中文原句：{{sentenceZh}}
用户输入：{{userInput}}

请从以下维度评分（1-10）并给出简短说明：
1. 语法正确性
2. 词汇选择
3. 表达地道性
4. 场景适合度
5. 礼貌性
6. 整体评分（1-100）

输出 JSON 格式，包含 suggestions 改进建议数组。
```

### 5.2 候选句子

```
给出中文句子的 3-5 种不同英文表达方式。
场景：{{sceneName}} - {{taskName}}
中文：{{sentenceZh}}

输出 JSON 数组，每项包含 sentence, style(polite/neutral/casual), note。
```

---

## 六、核心组件

| 组件 | 功能 |
|------|------|
| `SceneCard.vue` | 场景卡片，含进度 |
| `SentenceDisplay.vue` | 显示句子和关键词 |
| `PracticeInput.vue` | 练习输入和提交 |
| `AIFeedback.vue` | AI 点评展示 |
| `HintButton.vue` | 获取提示候选句 |
| `FlashCard.vue` | 闪卡翻转复习 |
| `StatsCard.vue` | 统计数字卡片 |

---

## 七、实现优先级

### Phase 1: 基础架构 (1-2天)
1. 初始化 Nuxt4 项目
2. 配置 Nuxt UI + Tailwind
3. 创建数据库初始化脚本
4. 创建基础 layouts

### Phase 2: 用户系统 (1天)
1. 邮箱注册/登录
2. Session 管理
3. 用户资料页面

### Phase 3: 核心练习 (2-3天)
1. 场景/任务列表页
2. 句子展示组件
3. OpenAI 集成
4. AI 点评流程

### Phase 4: 提示系统 (0.5天)
1. 候选句子 API
2. 提示展示组件

### Phase 5: 复习系统 (1天)
1. 闪卡组件
2. 复习设置
3. 播放功能

### Phase 6: 统计 (0.5天)
1. 学习统计 API
2. 统计展示页面

---

## 八、数据初始化

### 预置场景（8个）

| 场景 | 任务 | 句子数 |
|------|------|--------|
| 机场 | 值机、行李、安检、登机 | 12 |
| 酒店 | 入住、客房服务、退房 | 10 |
| 打车 | 叫车、说明目的地 | 6 |
| 餐厅 | 订座、点餐、结账 | 10 |
| 购物 | 问价、试穿、付款 | 10 |
| 工作 | 会议、邮件、汇报 | 12 |
| 看病 | 挂号、描述症状 | 8 |
| 问路 | 问方向、问距离 | 6 |

**总计**: MVP 约 74 个句子

---

## 九、验证方式

1. **本地运行**: `pnpm dev` 启动应用
2. **注册账号**: 测试邮箱注册登录
3. **场景练习**: 选择一个场景，练习句子
4. **AI 点评**: 提交句子，查看 AI 点评是否合理
5. **提示功能**: 点击提示，查看候选句子
6. **闪卡复习**: 进入复习模式，测试翻转和播放
7. **统计验证**: 练习后查看统计数据是否正确
8. **移动端**: 用手机访问，检查自适应效果

---

## 十、关键文件路径

- `prisma/schema.prisma` - Prisma 数据模型
- `prisma/seed.ts` - 初始数据种子
- `server/database/index.ts` - 数据库初始化
- `server/utils/openai.ts` - OpenAI 封装
- `server/api/auth/*.ts` - 认证 API
- `server/api/scenes/*.ts` - 场景 API
- `server/api/practice/*.ts` - 练习 API
- `server/api/ai/*.ts` - AI 点评 API
- `app/pages/*.vue` - 页面组件
- `app/components/*.vue` - UI 组件
- `app/layouts/*.vue` - 布局组件
- `app/composables/*.ts` - 组合式函数