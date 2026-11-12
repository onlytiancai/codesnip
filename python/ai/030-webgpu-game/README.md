# 序列化星空 — WebGPU 宇宙探索自由飞行游戏

> **第一人称飞船视角 · 自由飞行 · 区块化程序宇宙 · 位置持久化**
> WASD + 鼠标控制方向，输入坐标可瞬间传送。任何位置每次去都看到完全一样的天体（固定种子）。

![截图](screenshots/mvp7-persist.png)

## 核心特性

- **自由飞行**：PointerLockControls + WASD + Space/Ctrl + Shift 加速 ×6
- **无限宇宙**：按 (chunkX, chunkY, chunkZ) 区块化生成，每块含 1~3 颗恒星 + 行星 / 小行星带 / 彗星
- **种子固定**：`UNIVERSE_SEED = 0xc0ffee42n`，每次生成同一坐标 → 同一星系
- **位置持久化**：IndexedDB 存 player state（位置 + 四元数），关闭浏览器再开还在原地
- **传送**：按 T 打开传送对话框，输入 `(x, y, z)` 浮点坐标瞬移
- **重生点**：传送过的位置自动记录在重生点面板
- **WebGPU 优先 + WebGL2 降级**：复用 027 样板

## 控制

| 按键 | 作用 |
|---|---|
| 鼠标左键 click | 进入 pointer lock |
| **WASD / 方向键** | 平移 |
| **Space / Ctrl** | 上升 / 下降 |
| **Shift** | 加速 ×6 |
| **Mouse** | 视角（yaw / pitch） |
| **T** | 打开传送对话框 |
| **Esc** | 释放鼠标 |

## 坐标

游戏用**世界浮点坐标** `(x, y, z)`，单位 1 ly ≈ 1000 scene unit。

```
当前坐标: (1080, 0, 0)
所在区块: (1, 0, 0)
```

按 T 输入新坐标可瞬移过去。程序自动重新加载周围区块。

## 技术栈

- React 19 + TypeScript 5.9 + Vite 8
- three@0.185.1（WebGPURenderer + TSL）
- IndexedDB（player state + 重生点）
- 零运行时依赖（除 three / react）

## 开发

```bash
pnpm install
pnpm dev          # http://localhost:5173/
pnpm build
```

浏览器要求 WebGPU（Chrome 113+ / Edge 113+ / Safari 17.4+）。否则自动降级 WebGL2。URL 加 `?forceWebGL=1` 强制 WebGL。

## 架构

### 程序化宇宙

每个 chunk = 1000×1000×1000 scene unit 的 box，种子由 `hashChunk(cx,cy,cz) ^ UNIVERSE_SEED` 生成。

每个 chunk 含：
- 1~3 颗恒星（OBAFGKM 7 类加权抽样，互相距离 ≥ 220）
- 每颗恒星 3~8 颗行星（按距离分段：lava / rocky / water / desert / ice / gasGiant / iceGiant / dwarf）
- 50% 概率 1 条小行星带（InstancedMesh 80~200 颗）
- 0~3 颗彗星（Kepler 极扁椭圆轨道）

飞船周围 ACTIVE_CHUNK_RADIUS = 2 → 5×5×5 = 125 chunks 持续激活。

### Float32 精度

playerGroup 是飞船世界锚点。所有 chunk mesh 挂在 playerGroup 下，position = `chunkWorld - playerPos`（相对偏移在 ±5000 范围内）。无论飞船飞多远都不会精度抖动。

### 渲染

- 星点背景：5k InstancedMesh（远景，独立于 chunk）
- Chunk 内容：恒星 SphereBasic + halo Sprite；行星 InstancedMesh（按 type 复用材质）；小行星带 InstancedMesh；彗星核心 + 尾迹 Line
- 单 system < 30 draw calls
- FPS 60（M4 / Chromium）

### 持久化

- `playerState` store（v2）：位置 + 四元数 + 时间戳，debounce 5s 写入 + pagehide flush
- `history` store（v1 兼容）：传送过的位置（重生点）

### 复用与致谢

- `rendererFactory.ts` / `StarField.ts` / `ProceduralTextures.ts` / `tween.ts` 复用自兄弟项目 [`027-solar-system`](../027-solar-system)
- 第一人称控制：`three/addons/controls/PointerLockControls.js`
- 种子化：splitmix32 + MurmurHash 风格 32-bit hash