# ☀️ 太阳系探索 — 少儿科普 3D 演示

面向 8~14 岁儿童的 **Web 3D 太阳系交互演示**。打开即玩、无需说明、视觉风格友好，支持 WebGPU（自动降级 WebGL）。

> **在线体验**：本地 `pnpm run dev` 启动后访问 http://localhost:5173/

## ✨ 功能亮点

- 🌍 **太阳 + 8 大行星 + 月球**，全部用 canvas 程序化纹理生成（木星大红斑、土星环卡西尼缝、地球海洋/大陆/云层、火星极冠、海王星暗斑…）
- 🪐 **点击 / 搜索 / 导航栏** 任一天体 → 相机平滑聚焦 + 跟随 + 信息面板（真实数据 + 儿童科普 + 小知识）
- ⏸️ **暂停 / 0.1×~1000× 调速** / 重置全景 / 自动旋转
- 🎨 **三种显示模式**：探索 / 科普（标签+数据）/ 轨道（行星变暗，轨道突出）
- ⚙️ **14 项设置**：显示 / 动画 / 视觉 / 科普 / 音效，全部 localStorage 持久化
- 🌌 **5000 颗星星 + 5 个星云**，音效默认关闭，用户主动开启
- 📱 **响应式**：PC / 平板 / 手机（390×844 起 UI 不遮挡场景）

## 🚀 启动

```bash
pnpm install        # 或 npm install
pnpm run dev        # 开发模式 → http://localhost:5173/
pnpm run build      # 生产构建（tsc + vite）
pnpm run preview    # 预览生产构建
```

**调试参数**（URL）：
- `?forceWebGL=1` — 强制 WebGL2 路径（即使支持 WebGPU）
- `?nobloom=1` — 禁用 bloom 后处理（现已默认关闭）

## 🛠 技术栈

| 层 | 选择 | 说明 |
|---|---|---|
| 框架 | **React 19 + TypeScript 5.9 + Vite 8** | 命令式 SceneManager 编排 3D 场景，UI 全中文 |
| 渲染 | **three.js 0.185.1**（精确 pin） | `three/webgpu` 构建 + Vite 别名避免双份核心 |
| 后处理 | **RenderPipeline + BloomNode** | r185 TSL 函数式 API，默认关闭（避免驱动差异导致模糊） |
| 控件 | three/addons `OrbitControls` + `CSS2DRenderer` | 拖拽/缩放/平移 + DOM 标签层 |
| 状态 | Context + useReducer + localStorage | 设置项持久化 |
| 工具 | 自写 `tween.ts` (30 行缓动) | 不引入 GSAP / Tween.js |

**目录结构**：

```
src/
├── components/   TopBar · PlanetNav · ControlBar · InfoPanel · SettingsPanel
│                 SearchBox · FunFactCard · LoadingScreen · WelcomeHint · SolarSystemCanvas
├── scene/        SceneManager · rendererFactory · buildSolarSystem · ProceduralTextures
│                 CameraController · Picking · starField
├── data/         planetData.ts（NASA 真实数据）· uiText.ts（儿童文案）
├── utils/        scale · format · tween · performance · audio
└── state/        AppState.tsx
```

## 🌌 WebGPU 与降级

r185 的 `WebGPURenderer` 已内置 WebGL2 自动降级。本项目统一走 `three/webgpu` 单一构建：

```
three/webgpu
  ↓
WebGPURenderer({ forceWebGL })  // 自动尝试 WebGPU → 失败降级 WebGL2
  ↓
renderer.backend.isWebGPUBackend  // 检测实际后端
```

**三层防御**（`rendererFactory.ts`）：

1. URL 参数 `?forceWebGL=1` 或 `navigator.gpu` 不存在 → 直接 `forceWebGL: true`
2. `await renderer.init()` 抛错 → dispose 后强制 WebGL2 重建
3. 首帧 `render()` 抛错（驱动缺陷）→ 同样重建
4. 两个后端都失败 → 友好错误页（🛸 飞船启动失败 + 重试按钮）

降级时顶栏徽章显示「兼容模式」+ toast 提示。

## 📐 真实数据 vs 视觉比例

**严格分离** —— 修改任意一端都不会影响另一端：

```
data/planetData.ts          ← NASA Fact Sheet 真实数据（半径 km / AU / 周期天 / 轴倾角…）
           ↓
utils/scale.ts              ← 视觉换算集中此处（距离压缩、半径封顶、周期相对化）
           ↓
scene/buildSolarSystem.ts   ← 仅消费最终视觉单位
```

**视觉压缩策略**：

| 维度 | 处理 |
|---|---|
| 轨道距离 | `au^0.55` 压缩到 6~45 单位：内行星密集、火木间保留明显空隙（10.4→18.4）、外行星遥远 |
| 行星半径 | 真实比例开平方 + 可点击下限 0.35：木星 3.35 > 土星 3.02 > 天王 2.0 ≈ 海王 1.97 > 地球 1.0 ≈ 金星 0.97 > 火星 0.73 > 水星 0.62，顺序与真实一致 |
| 太阳 | 真实 109 倍地球 → 视觉封顶 4.0（必须小于水星轨道 6.0，否则被吞没） |
| 公转周期 | `天^0.6`：水星 8s/圈 → 海王星 400s/圈，相对快慢真实，1000× 下外行星可见 |
| 自转 | 真实周期比例：木星 1.3s 最快、金星 31s 逆向、天王星轴倾角 97.8° 躺着转 |
| 月球 | 真实 60 倍地球半径 → 视觉 2.2 单位 |

## 🐛 关键问题修复记录

开发过程中遇到的最棘手的几个 bug：

1. **「卡在启动画面」的真正元凶**：`earthCloudTexture` 和 `saturnTexture` 里 `fbmWarp(rng(44), ...)` 在 fbm2 回调内**每像素新建一个 rng**，导致 WeakMap 缓存每次都 miss，131k 像素 × 完整 fbm 计算 → 数十亿次运算卡死 JS 主线程。
2. **React settings effect 在 init 完成前触发场景 API** → 异常 → React 树卸载 → dispose 销毁 WebGPU 渲染器 → GPU 卡死整页。已加 init 守卫。
3. **backdrop-filter 叠在 60fps canvas 上** → 合成器 CPU 100%。已移除。
4. **WebGPURenderer 不支持 `LineLoop`** → 改用闭合 `Line`（首尾重复点）。
5. **bloom 渲染管线在某些驱动下产生大模糊半径泛光** → 占据整个屏幕让行星看起来发糊。已**默认关闭 bloom**，太阳光感改为纯 `MeshBasicMaterial({ toneMapped: false })` + emissive。

## ✅ 验收清单

### 功能

- [x] 八大行星全部存在
- [x] 太阳存在
- [x] 行星顺序正确（实测 ✓）
- [x] 行星围绕太阳运行
- [x] 行星可以自转
- [x] 地球有月球
- [x] 土星有明显行星环 + 卡西尼缝
- [x] 木星有明显大红斑
- [x] 地球有明显海洋 / 陆地 / 云层
- [x] 行星可以点击
- [x] 点击后显示信息
- [x] 相机可以聚焦行星 + 跟随
- [x] 可以旋转视角
- [x] 可以缩放
- [x] 可以平移
- [x] 可以暂停
- [x] 可以调整时间速度（0.1×~1000×）
- [x] 可以重置视角
- [x] 有设置面板
- [x] 有轨道显示开关
- [x] 有标签显示开关
- [x] WebGPU 可用时优先使用 WebGPU（实测 ✓ 顶栏显示「⚡ WebGPU」）
- [x] WebGPU 不支持时可以 WebGL fallback
- [x] 手机端基本可用（响应式断点 + 移动端布局）

### 科学性

- [x] 行星大小关系合理（木星最大，水星最小）
- [x] 行星距离关系合理（火木空隙 > 内行星间距）
- [x] 行星轨道顺序正确
- [x] 数据来源可靠（NASA Planetary Fact Sheet）
- [x] 不虚构科学数据
- [x] 明确区分真实比例和视觉展示比例（`scale.ts` 注释说明）

### UX

- [x] 第一屏就能看到太阳系
- [x] 不需要复杂配置才能开始
- [x] 小朋友能够理解（图标 + emoji + 大按钮 + 简短文案）
- [x] UI 不遮挡主要 3D 场景
- [x] 移动端可以正常使用
- [x] Loading 状态完善
- [x] 错误处理完善

## 🎯 可进一步优化

- 主 chunk 1.1MB（gzip 318KB，three.js WebGPU 构建固有体积）—— 可改为 CDN 引入或精细分包
- 行星选中态用 outline / 描边替代 emissive（已实现）；可加拖尾残影
- 距离比较互动（地球 → 木星相距……）
- 全屏截图分享按钮
- 设置面板开关的自动化回归测试
- 移动端端到端测试（390×4 + 触控）

## 📝 许可

仅作少儿科普演示用途。行星纹理使用 canvas 程序化生成，无版权争议。`three.js` 使用 MIT 协议。