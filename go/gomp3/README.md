# gomp3

一个基于 Go 和 [MyGo](https://github.com/egoist/mygo) v0.7.4 的 macOS 原生 MP3 音效播放器。

音频解码与全部音效处理均为纯 Go 实现（`CGO_ENABLED=0`），输出走 MyGo 的 `audio` 包（macOS 上通过 purego 调用 CoreAudio），无需 CGO、无需虚拟声卡。界面包含基础、高级、专家、3D 四个页签，提供从三段旋钮到参数均衡器的完整音效工作台。

## 功能特性

**播放**
- 打开本地 MP3 文件并播放，支持播放 / 暂停 / 停止、拖动进度条跳转
- 接入 macOS 媒体键与「正在播放」（Now Playing）信息
- 实时显示文件名与元数据、播放进度

**均衡器（四个页签）**
- **基础**：低音 / 中音 / 高音三段旋钮（±12 dB）
- **高级**：固定 10 段图形 EQ（32 Hz – 16 kHz）与固定 31 段图形 EQ（20 Hz – 20 kHz，1/3 倍频程，横向滚动），两者互斥切换、独立开关；带合成频响曲线
- **专家**：动态参数均衡器，可任意增删波段；每波段支持 7 种滤波器类型（峰值、低频搁架、高频搁架、低通、高通、带通、陷波），可调频率、增益、Q 值或带宽（按 RBJ 公式互换）
- **3D**：虚拟环绕声，可调低频效果与环绕强度

**音效处理**
- 前置增益（Preamp，±12 dB）
- 音量增强 100% – 200%（带软削波与峰值限制器）
- 左右平衡调节与单声道开关
- 峰值限制器（1 ms attack / 100 ms release），防止输出削波

**可视化**
- 实时 FFT 频谱分析：2048 点 Hann 窗、256 个对数间隔频段、峰值保持
- 与各 EQ 页联动的频响曲线绘制

**预设**
- 10 个内置预设：Flat、Pop、Classical、Movie、Hip Hop、Jazz、Electronic、Rock、Bass Boost、Vocal
- 自定义预设保存 / 删除，JSON 本地持久化（原子写入）

> 音量增强与 3D 环绕均只在应用自身音频链内生效，不会修改 macOS 系统音量，也不需要虚拟声卡或额外权限。

## 环境要求

- macOS 12.0 或更高版本
- Go 1.27.3（项目通过 `go.mod` 的 `tool` 指令内置 MyGo 命令行工具）
- 现有打包产物为 Apple Silicon（arm64）；源码可在其他平台编译，但签名与 DMG 生成必须在 macOS 上完成

主要依赖：MyGo v0.7.4（UI 与音频输出）、go-mp3 v0.3.4（MP3 解码）、purego v0.11.1（无 CGO 的系统调用）。

## 快速开始

```bash
cd /Users/huhao/src/codesnip/go/gomp3

# 下载依赖
go mod download

# 开发运行（live reload）
go tool mygo dev

# 生产打包：生成 .app 与 .dmg 到 build/darwin-arm64/
go tool mygo build
```

打包产物：
- `build/darwin-arm64/gomp3.app`
- `build/darwin-arm64/gomp3-0.1.0-arm64.dmg`

## 测试与检查

```bash
# 单元测试
CGO_ENABLED=0 go test ./...

# 竞态检测
CGO_ENABLED=0 go test -race ./...

# MyGo 生命周期 / UI 规则检查
go tool mygo vet .

# 重新生成 docs/ 下的界面截图（默认跳过）
RENDER_PNG=1 go test -run TestScreenshot ./...
```

> `go tool mygo build` 生成 DMG 时偶发因 `/Volumes/gomp3` 挂载点清理冲突失败，重试一次即可。

## 配置与持久化

应用配置以 JSON 保存在：

```
~/Library/Application Support/gomp3/config.json
```

可通过环境变量 `GOMP3_CONFIG` 指定其他路径（测试时使用）。写入采用临时文件 + 原子重命名，避免损坏。保存内容包括：当前激活预设、自定义预设、上次打开的文件、当前 EQ 页签、10/31 段模式。

## 项目结构

```
.
├── main.go              # 应用装配、回调（打开/播放/停止/跳转/音量/预设 CRUD）、UI 同步 ticker
├── player.go            # 播放器封装：解码器 + DSP + MyGo Player，媒体键集成
├── presets.go           # 内置/自定义预设、配置加载与原子保存
├── main_test.go         # UI 交互测试（专家页添加波段、保存预设）
├── presets_test.go      # 预设与配置持久化测试
├── screenshot_test.go   # 界面截图生成（RENDER_PNG=1 时启用）
├── go.mod / go.sum      # 依赖与 mygo 工具声明
├── mygo.json            # 应用名、标识、版本、输出目录
├── dsp/                 # 纯 Go 音频处理
│   ├── config.go        # 滤波器类型、10/31 段频点、Pipeline 配置与快照
│   ├── biquad.go        # RBJ 双二阶滤波器系数设计与处理、频响计算
│   ├── eq.go            # 滤波器链级联、左右声道独立状态、合成频响
│   ├── gain.go          # 峰值限制器、软削波、左右平衡
│   ├── surround.go      # 虚拟 3D 环绕（低频搁架 + M/S 展宽 + 交叉延时）
│   ├── fft.go           # FFT 频谱分析、对数频段、峰值保持
│   ├── decoder.go       # Decoder 包装：信号链编排、DSP 快照发布、seek 复位
│   └── dsp_test.go      # DSP 单元测试
└── ui/                  # MyGo 原生界面
    ├── view.go          # 根布局、工具栏、进度、基础/高级/3D 页、预设弹窗
    ├── knobs.go         # 自绘旋钮
    ├── faders.go        # 10/31 段垂直推子
    ├── graph.go         # 频响曲线绘制
    ├── spectrum.go      # 实时频谱绘制
    ├── expert.go        # 参数 EQ 波段表格
    ├── model.go         # UI 状态与回调模型
    └── theme.go         # 配色主题
```

## 音频处理链

```
MP3 解码 → 前置增益 → EQ 滤波器链 → 3D 环绕 → 平衡 / 音量增强 → 峰值限制 → 频谱分析 → 输出
```

- 全部处理在 `dsp.Decoder.Read()` 中按缓冲区顺序执行，左右声道各自维护滤波器状态。
- 配置通过 `atomic.Pointer[Pipeline]` 以快照方式在音频线程与 UI 线程间传递，无需加锁。
- 调整音效后通过 `Seek(Position())` 刷新预取缓冲，使改动即时生效；跳转时复位 DSP 状态。

## 备注

界面截图由 `screenshot_test.go` 生成（见「测试与检查」），属生成产物，未纳入版本管理。
