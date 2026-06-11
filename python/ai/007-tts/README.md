# 007-tts · Kokoro TTS Web Demo

基于 [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) 的多语言、多音色文本转语音 Web Demo，使用 FastAPI 提供后端 + 静态前端页面。

> 参考原命令式脚本：[1.py](1.py)。

---

## ✨ 功能

- 🌍 **9 种语言 / 32 个精选音色**（中文 / 美式英语 / 英式英语 / 日语 / 西语 / 法语 / 印地语 / 意语 / 巴西葡语）
- 🎚️ **语速可调** 0.5× – 2.0×
- 📦 **WAV / MP3 双格式**（MP3 由 `ffmpeg -c:a libmp3lame -q:a 2` 编码）
- 🔊 **Web 页面**：文本输入、音色网格（搜索 + 语言筛选）、内置播放器、下载、⌘/Ctrl+Enter 合成
- 🗂️ **历史记录**：自动存入 `localStorage`，一键回填参数
- 🧠 **懒加载 KPipeline**：按 `lang_code` 缓存模型，节省显存
- 📁 **落盘归档**：每次合成都额外写到 `outputs/`，可重复下载

---

## 📂 目录结构

```
007-tts/
├── 1.py                  # 原始 CLI 脚本（参考）
├── app.py                # FastAPI 后端
├── tts_service.py        # Kokoro 封装（懒加载、合成、ffmpeg 转码）
├── requirements.txt
├── run.sh                # 一键启动（自动 pyenv activate qlib）
├── outputs/              # 合成产物自动归档
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

---

## 🚀 启动

```bash
# 推荐：使用项目自带的启动脚本（自动切换到 pyenv qlib 环境）
cd 007-tts
./run.sh

# 或手动：
pyenv activate qlib
python -m uvicorn app:app --host 127.0.0.1 --port 8765
```

打开浏览器访问 [http://127.0.0.1:8765](http://127.0.0.1:8765) 即可。

> 首次合成中文/英文会下载 `Kokoro-82M` 模型权重到 `~/.cache/huggingface/`；英文首次还会拉取 `en_core_web_sm` spaCy 模型。

---

## 🔌 API

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET`  | `/`                  | 静态前端页面 |
| `GET`  | `/api/health`        | 健康检查 |
| `GET`  | `/api/voices`        | 全部音色列表 |
| `GET`  | `/api/langs`         | 支持的语言列表 |
| `POST` | `/api/tts`           | 合成语音（JSON 请求体，返回音频二进制） |
| `GET`  | `/api/tts?text=&voice=&speed=&format=&download=` | 合成语音（URL 参数，适合 `<audio src>`） |
| `GET`  | `/api/outputs/{fname}` | 下载/试听历史产物 |
| `GET`  | `/docs`              | FastAPI 自动生成的 OpenAPI 文档 |

### `POST /api/tts` 请求体

```json
{
  "text": "今天我们来聊一个有趣的话题。",
  "voice": "zf_xiaoni",
  "speed": 1.0,
  "format": "wav"
}
```

- `voice`：音色名，前两位代表语言与性别（`z=中文 / a=美式 / b=英式 / j=日 / e=西 / f=法 / h=印 / i=意 / p=巴葡`，第二位 `f=女 / m=男`）。
- `speed`：0.5 – 2.0。
- `format`：`wav` | `mp3`（选 `mp3` 需要系统里有 `ffmpeg`）。

返回头包含 `Content-Type`、`Content-Disposition`、`X-File-Name`（产物归档名）。

### curl 示例

```bash
# 中文
curl -X POST http://127.0.0.1:8765/api/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"你好，这是一个测试。","voice":"zf_xiaoni","format":"wav"}' \
  -o out.wav

# 英文（GET 形式）
curl "http://127.0.0.1:8765/api/tts?text=Hello%20world&voice=af_heart&format=mp3" \
  -o out.mp3
```

---

## 🎨 音色速查

### 🇨🇳 中文 (`lang_code='z'`)

| Name | 性别 | 标签 |
| --- | --- | --- |
| `zf_xiaobei`  | 女 | 小蓓 |
| `zf_xiaoni`   | 女 | 小妮 |
| `zf_xiaoxiao` | 女 | 小晓 |
| `zf_xiaoyi`   | 女 | 小怡 |
| `zm_yunjian`  | 男 | 云健 |
| `zm_yunxi`    | 男 | 云溪 |
| `zm_yunxia`   | 男 | 云霞 |
| `zm_yunyang`  | 男 | 云扬 |

### 🇺🇸 美式英语 (`a`)

`af_heart`(A) · `af_bella`(A-) · `af_nicole`(B-) · `am_adam` · `am_michael`(B) · `am_puck`(B)

### 🇬🇧 英式英语 (`b`)

`bf_emma`(B-) · `bf_isabella`(B) · `bm_george`(B) · `bm_lewis`(C)

> 更多 / 完整音色表见 [Kokoro 官方 VOICES.md](https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)。

---

## 🔁 与原 `1.py` 的关系

`1.py` 是命令行的最小验证脚本。本 Web Demo 的核心合成逻辑与之一致（`KPipeline` → `voice=...` → 写 WAV），并在此基础上：

1. 拆出 `tts_service.py` 做懒加载与多格式输出；
2. 加 FastAPI 接口（`POST / GET`）；
3. 加静态前端（`static/`）做可视化试听；
4. 每次合成自动归档到 `outputs/` 方便下载；
5. ffmpeg 可选启用，缺省时自动回退到 WAV。

---

## 🧯 常见问题

- **首次合成很慢**：Kokoro 在按需下载模型 + 初始化 `KPipeline`，后续调用会快很多（毫秒级）。
- **MP3 失败但 WAV 成功**：系统里没装 `ffmpeg` 或不在 `PATH`。装一下 `brew install ffmpeg` 即可。
- **英文报 `Can't find model 'en_core_web_sm'`**：第一次英文合成会自动 `pip install` spaCy 模型；如离线请提前 `python -m spacy download en_core_web_sm`。
- **想自定义音色库**：编辑 [tts_service.py](tts_service.py) 顶部的 `VOICE_LIBRARY` 即可。

---

## 🧪 验证记录

| 场景 | 结果 |
| --- | --- |
| `/api/health` | `{"status":"ok"}` |
| `/api/voices` | 32 个音色 |
| 中文 `zf_xiaoni` | 174 KB WAV，16-bit / 24 kHz / mono |
| 英文 `af_heart` | 114 KB WAV |
| 男声 `zm_yunxi` | MP3（libmp3lame） |
| 静态首页 / CSS / JS | 全部 200 |
