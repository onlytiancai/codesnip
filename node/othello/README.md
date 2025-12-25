# 奥赛罗棋（黑白翻转棋）在线多人对战游戏

## 项目概述

这是一个基于Web的奥赛罗棋（也称为黑白翻转棋）游戏，支持在线多人对战。游戏采用前后端分离架构，使用WebSocket实现实时通信，确保游戏状态的同步和公平性。

## 功能特性

### 基础游戏规则
- 8×8标准棋盘
- 黑棋先手，白棋后手
- 严格实现落子与翻转规则（8个方向）
- 自动校验合法落子
- 无子可下时自动跳过回合
- 游戏结束自动统计胜负

### 在线多人对战
- 房间创建/加入功能（通过房间ID）
- 每个房间最多支持2名玩家+观战者
- WebSocket实时同步棋盘状态、当前玩家、棋子颜色和游戏结束状态
- 后端作为权威状态源，确保游戏公平性

### 前端功能
- Vue 3 Composition API实现响应式界面
- Tailwind CSS构建美观的棋盘和UI
- 棋盘交互与有效落子高亮显示
- 显示当前玩家、棋子数量和游戏结果
- 断线重连处理

### 后端功能
- Node.js + WebSocket管理房间和玩家
- 清晰的数据结构（Room、Player、GameState）
- 后端校验所有落子并广播结果
- 支持多房间并发

## 技术栈

### 前端
- **框架**: Vue 3 (Composition API)
- **构建工具**: Vite
- **样式**: Tailwind CSS 3
- **语言**: TypeScript
- **实时通信**: WebSocket API

### 后端
- **运行环境**: Node.js
- **Web框架**: Express
- **实时通信**: ws（WebSocket库）
- **语言**: JavaScript

## 项目结构

```
othello/
├── client/                  # 前端项目
│   ├── public/              # 静态资源
│   ├── src/                 # 源代码
│   │   ├── components/      # Vue组件
│   │   │   └── Board.vue    # 棋盘组件
│   │   ├── App.vue          # 主应用组件
│   │   ├── main.ts          # 应用入口
│   │   ├── style.css        # 全局样式
│   │   └── counter.ts       # 计数器示例（可删除）
│   ├── index.html           # HTML模板
│   ├── package.json         # 前端依赖
│   ├── pnpm-lock.yaml       # 依赖锁定文件
│   ├── postcss.config.js    # PostCSS配置
│   ├── tailwind.config.js   # Tailwind CSS配置
│   └── tsconfig.json        # TypeScript配置
├── server/                  # 后端项目
│   ├── src/                 # 源代码
│   │   ├── game/            # 游戏相关模块
│   │   │   ├── game.js      # 游戏逻辑
│   │   │   ├── player.js    # 玩家模型
│   │   │   ├── room.js      # 房间模型
│   │   │   └── roomManager.js # 房间管理器
│   │   ├── index.js         # 后端入口
│   │   └── websocket.js     # WebSocket处理
│   ├── package.json         # 后端依赖
│   └── pnpm-lock.yaml       # 依赖锁定文件
├── WEBSOCKET_PROTOCOL.md    # WebSocket协议文档
└── README.md                # 项目说明文档
```

## 安装与设置

### 前置要求
- Node.js 16+ 
- pnpm（推荐）或npm/yarn

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <项目地址>
   cd othello
   ```

2. **安装前端依赖**
   ```bash
   cd client
   pnpm install
   cd ..
   ```

3. **安装后端依赖**
   ```bash
   cd server
   pnpm install
   cd ..
   ```

## 运行项目

### 启动后端服务器
```bash
cd server
pnpm run start
# 服务器将在 http://localhost:3001 运行
```

### 启动前端开发服务器
```bash
cd client
pnpm run dev
# 前端将在 http://localhost:5173 运行（端口可能因占用而变化）
```

### 构建生产版本
```bash
cd client
pnpm run build
# 构建产物将在 dist 目录
```

## WebSocket协议

详细的WebSocket协议定义请参考 `WEBSOCKET_PROTOCOL.md` 文件。

主要消息类型：
- `CREATE_ROOM` - 创建新房间
- `JOIN_ROOM` - 加入现有房间
- `MAKE_MOVE` - 落子
- `ROOM_UPDATED` - 房间状态更新
- `MOVE_MADE` - 落子结果
- `ERROR` - 错误信息

## 游戏规则

1. 游戏在8×8的棋盘上进行，黑棋先手
2. 玩家必须将自己的棋子放在棋盘上，使得至少有一个对方棋子被自己的棋子夹在中间
3. 被夹住的对方棋子将翻转成己方颜色
4. 如果玩家没有合法的落子位置，将跳过该回合
5. 当双方都无法落子时，游戏结束
6. 棋盘上棋子较多的一方获胜，若数量相同则为平局

## 代码扩展

项目代码结构清晰，易于扩展：

- **AI功能**: 可在 `server/src/game/` 目录下添加AI相关代码
- **排行榜**: 可添加数据库支持和排行榜API
- **游戏记录**: 可实现游戏回放功能
- **更多棋类**: 可基于现有架构扩展其他棋类游戏

## 开发说明

### 前端开发
- 使用Vue 3 Composition API进行组件开发
- Tailwind CSS类名用于样式设计
- WebSocket客户端在App.vue中实现

### 后端开发
- 房间管理：`roomManager.js` 负责房间的创建、加入和删除
- 游戏逻辑：`game.js` 实现奥赛罗棋的核心规则
- WebSocket处理：`websocket.js` 处理客户端消息

## 许可证

ISC

## 联系方式

如有问题或建议，请联系项目维护者。