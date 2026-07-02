# XOR 反向传播 · 推导讲解视频

把 `006_02.ipynb` 里"为什么 ∂L/∂z = ŷ − y"的 4 步链式法则推导,做成给数学/ML 基础薄弱观众看的讲解视频。

> **核心目标**:让观众从 sigmoid + 交叉熵的反向求导过程中,**亲眼看到** `dZ2 = ŷ − y` 是怎么"凑巧"出现的,然后映射到一行 NumPy 代码。

---

## 项目结构

```
/013/
├── README.md                       ← 本文件
├── desc.json                       ⭐ 视频脚本(待审核,12 张卡)
├── pipeline/                       ← Python 图表生成 + TTS
│   ├── render_assets.py
│   ├── render_audio.py
│   ├── requirements.txt
│   └── output/
│       ├── formulas/               ← 公式 PNG
│       ├── plots/                  ← 曲线 PNG (sigmoid / 交叉熵 / 导数)
│       └── diagrams/               ← 网络结构 PNG
└── video/                          ← Remotion 端
    ├── package.json                ← pnpm
    ├── remotion.config.ts
    ├── tsconfig.json
    ├── public/
    │   ├── images/                 ← 从 pipeline/output/ 拷贝
    │   └── audio/                  ← TTS 产物(.mp3)
    └── src/
        ├── index.ts
        ├── Root.tsx
        ├── Video.tsx               ← 主组合
        ├── theme.ts                ← paper 主题 + 横版 LAYOUT
        ├── hooks/useFadeIn.ts
        └── components/
            ├── Header.tsx
            ├── Footer.tsx
            └── cards/
                ├── IntroCard.tsx
                ├── FormulaCard.tsx  (formula/math_anim/plot/diagram)
                └── CodeCard.tsx
```

---

## 视频分集

画幅: **1920×1080 横版** · 30 fps · 中文旁白 · `paper` 白底主题

| # | type | 标题 | 时长 |
|---|------|------|-----|
| 0 | intro | 开场:为什么需要这次推导 | ~10s |
| 1 | diagram | 网络结构 (2→4→1) | ~12s |
| 2 | plot | sigmoid 函数 | ~10s |
| 3 | plot | 交叉熵损失 | ~12s |
| 4 | formula | 链式法则:拆成两步 | ~10s |
| 5 | formula | 求 ∂L/∂ŷ | ~15s |
| 6 | formula | 化简 ∂L/∂ŷ | ~12s |
| 7 | plot | sigmoid 导数 | ~12s |
| 8 | math_anim | **高潮:两式相乘 → ŷ−y** | ~20s |
| 9 | code | 映射到 NumPy | ~10s |
| 10 | code | 彩蛋:3 行反传代码 | ~8s |

---

## 完整运行流程(用户审 desc.json 后)

### 1. 安装依赖

```bash
# Python
/Users/huhao/.pyenv/versions/3.11.9/bin/pip install -r pipeline/requirements.txt

# Remotion
cd video
pnpm install
```

### 2. 生成所有图片(公式 + 曲线 + 网络)

```bash
cd /013
/Users/huhao/.pyenv/versions/3.11.9/bin/python pipeline/render_assets.py
```

产物:
- `pipeline/output/diagrams/01-network.png`
- `pipeline/output/plots/02-sigmoid.png`, `03-crossentropy.png`, `07-sigmoid-deriv.png`
- `pipeline/output/formulas/04-chain.png`, `05-dldyhat.png`, `06-simplify.png`
- `pipeline/output/formulas/08-step1..5.png`(高潮 5 步帧)

### 3. 生成所有 TTS 音频

```bash
export MINIMAX_API_KEY=你的_key
/Users/huhao/.pyenv/versions/3.11.9/bin/python pipeline/render_audio.py
```

每张卡 1 段中文音频 (MiniMax `speech-02-hd` / `female-shaonv`)。
脚本会自动回填 `desc.json` 里的 `audio_path` / `duration_ms` / `duration_sec` / `duration_frames`。

### 4. 拷贝资产到 Remotion public/

```bash
cd video
pnpm exec tsx scripts/render.ts --phase assets
```

### 5. 渲染最终视频

```bash
cd video
pnpm exec tsx scripts/render.ts --phase video
# 产物: video/out/xor-bp.mp4
```

或者一步全跑:

```bash
cd /013
/Users/huhao/.pyenv/versions/3.11.9/bin/python pipeline/render_assets.py
/Users/huhao/.pyenv/versions/3.11.9/bin/python pipeline/render_audio.py
cd video
pnpm exec tsx scripts/render.ts --phase all
```

### 6. 验证产物

```bash
ffprobe video/out/xor-bp.mp4
```

应看到:1920×1080 · 约 130s · h264 · ~30fps

---

## 字体

- **中文**:PingFang SC(苹方-简),存在于 `/System/Library/AssetsV2/.../PingFang.ttc`
- **数学**:matplotlib mathtext `cm`(Computer Modern 风格)

`pipeline/render_assets.py` 顶部已显式 `fontManager.addfont(...)` 注册,无需手动配置。

---

## 工作流备忘

- **用户审稿阶段**:改了 `desc.json` 里的旁白 → 重新跑 `render_audio.py`(只回填 duration) → 再 `render video`。
- **math_anim 帧**:5 张 PNG 已预渲染在 `pipeline/output/formulas/08-step{1..5}.png`,Remotion 端按 frame 做 cross-fade 衔接。
- **caption 字幕**:每张卡底部一行,≤18 字,与音频同步淡入 — 基础薄弱观众的"听+看"双通道。

---

## 关联

- 原推导教程:`../006_02.ipynb`
- 同类视频项目参考:`/Users/huhao/src/codesnip/node/en-sentence-study/scripts/video/(英语口语卡片,本项目的结构借鉴)
