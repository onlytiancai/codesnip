# MiniMax LLM & TTS API 使用参考

> 面向以后复用的小抄。包含：鉴权、两个端点 URL、请求/响应字段、易踩坑、Python 示例。
> 官方文档以 <https://platform.MiniMax.io/docs> 为准，本文以实测接口为准。

---

## 0. 鉴权与环境

所有接口走 **API Key**，单 header：

```
Authorization: Bearer $MINIMAX_API_KEY
Content-Type: application/json
```

`MINIMAX_API_KEY` 从 <https://platform.MiniMax.io/user-center/basic-information/interface-key> 获取。

**国内/海外网络不通时**，设置 HTTPS 代理：

```bash
export HTTPS_PROXY=http://127.0.0.1:10808
```

Python `urllib` 会自动读取该环境变量。

---

## 1. LLM：`chat/completions`（OpenAI 兼容）

### 1.1 端点

```
POST https://api.MiniMax.com/v1/chat/completions
```

### 1.2 请求

```json
{
  "model": "MiniMax-M3",
  "messages": [
    {"role": "system", "content": "You are …"},
    {"role": "user",   "content": "Hello"}
  ],
  "max_completion_tokens": 8192,
  "temperature": 0.7
}
```

| 字段 | 说明 |
| --- | --- |
| `model` | 模型 ID，常用 `MiniMax-M3`（推理强）、`MiniMax-Text-01` 等 |
| `messages` | OpenAI 风格多轮对话 |
| `max_completion_tokens` | 输出上限；任务复杂（剧本 JSON）建议 8192+ |
| `temperature` | 可选，默认 0.7；结构化输出可降到 0.3~0.5 |
| `stream` | 可选 `true`，流式；本文档只涉及非流式 |

### 1.3 响应

```json
{
  "id": "…",
  "model": "MiniMax-M3",
  "choices": [
    {
      "index": 0,
      "message": {"role": "assistant", "content": "…"},
      "finish_reason": "stop"
    }
  ],
  "usage": {"prompt_tokens": …, "completion_tokens": …, "total_tokens": …}
}
```

取 `choices[0].message.content`。

### 1.4 易踩坑

- **`<<think>>…</think>>` 前缀**：`MiniMax-M3` 等推理模型会先吐一段思维链到 `content` 里。
  解析 JSON 前必须用正则 `<think>.*?</think>` 剥离（`re.DOTALL`），否则 `json.loads` 必失败。
- **包 markdown 代码块**：偶发。提示词里明令「不要 ```json 包裹」可降低概率，但程序仍要做兜底：扫描首个 `{` 到匹配 `}`。
- **越界 token**：超长剧本会被截断；分段提交或提升 `max_completion_tokens`。

---

## 2. TTS：`t2a_v2`（文本 → 音频）

### 2.1 端点

```
POST https://api.MiniMax.com/v1/t2a_v2
```

### 2.2 请求

```json
{
  "model": "speech-02-hd",
  "text": "你好，世界。",
  "stream": false,
  "language_boost": "Chinese",
  "voice_setting": {
    "voice_id": "male-qn-qingse",
    "speed": 1.0,
    "vol": 1.0,
    "pitch": 0,
    "emotion": "calm"
  },
  "audio_setting": {
    "sample_rate": 32000,
    "bitrate": 128000,
    "format": "mp3",
    "channel": 1
  },
  "pronunciation_dict": {
    "tone": ["处理/(chu3)(li3)", "角色/(jue4)(se4)"]
  }
}
```

| 字段 | 取值 | 说明 |
| --- | --- | --- |
| `model` | `speech-02-hd` / `speech-02-turbo` / `speech-01` | 推荐 `speech-02-hd` 高保真 |
| `text` | string | 单次最长 ~1 万字符；超长需切片 |
| `stream` | bool | `false` = 一次性返回完整音频 hex |
| `language_boost` | `Chinese` / `English` / `Japanese` / … | 中文小说固定 `Chinese` |
| `voice_setting.voice_id` | 见 `voices.py` 音色池 | 系统音色 ID |
| `voice_setting.speed` | **0.5 – 2** | 1.0 = 正常 |
| `voice_setting.vol` | **0 – 10** | 1.0 = 正常 |
| `voice_setting.pitch` | **-12 – 12** | 半音；0 = 原调 |
| `voice_setting.emotion` | 见下方枚举 | 不填默认 neutral |
| `audio_setting.sample_rate` | 8000 / 16000 / 22050 / **32000** / 44100 | Hz |
| `audio_setting.bitrate` | 32000 / 64000 / **128000** / 256000 | bps（mp3 有效） |
| `audio_setting.format` | **mp3** / pcm / flac / wav | wav 时填 `pcm` 后改名 |
| `audio_setting.channel` | 1 = 单声道，2 = 立体声 | 单人/旁白用 1 |
| `pronunciation_dict.tone` | 字符串数组 | 全局多音字词典，作用于整段 text |

### 2.3 emotion 枚举（**严格**，越界会报错）

```
happy | sad | angry | fearful | disgusted | surprised | calm | fluent | whisper
```

> neutral / warm / cool / excited 等不在枚举中，禁止使用。

### 2.4 行内停顿与多音字

**停顿**（写在 `text` 中）：

```
今天天气真好<#1.5#>我们继续赶路吧。
```

`<#x#>` 表示停顿 **x 秒**（小数）；常用于段落留白。

**多音字两种写法**：

1. **行内注音**（推荐，单条生效）：
   ```
   这件事得(děi)马上去处理(chǔ lǐ)。
   ```
2. **全局词典**（作用于整段，多次出现统一读音）：
   ```json
   "pronunciation_dict": {"tone": ["处理/(chu3)(li3)"]}
   ```
   数字标声调：1=阴平 2=阳平 3=上声 4=去声 5=轻声，括号紧贴汉字。

### 2.5 响应

```json
{
  "data": {
    "audio": "ffffff7f7f7e…(hex)…",
    "status": 2
  },
  "extra_info": {
    "audio_length": 4523,
    "audio_sample_rate": 32000,
    "audio_size": 73421,
    "bitrate": 128000,
    "audio_format": "mp3",
    "audio_channel": 1,
    "invisible_character_meta": null,
    "usage_characters": 9
  },
  "base_resp": {
    "status_code": 0,
    "status_msg": "success"
  }
}
```

| 字段 | 说明 |
| --- | --- |
| `data.audio` | **hex 编码** 的二进制音频；有时也返回 base64，按 `extra_info.audio_format` 区分时通常 hex |
| `extra_info.audio_length` | **毫秒**（注意单位）；用于估算合成耗时与片段拼接 |
| `extra_info.usage_characters` | 实际计费字符数（含停顿标记） |
| `base_resp.status_code` | `0` = 成功；非 0 看 `status_msg` |

### 2.6 易踩坑

- **`data.audio` 是 hex 字符串**，要先 `bytes.fromhex(...)` 再落盘；不要直接 `utf-8` 解码。
- **`audio_length` 单位是毫秒**，不是秒；做 UI 显示要 `/1000`。
- **`emotion` 拼写错误**会直接 `status_code != 0`，程序侧要先校验枚举。
- **`speed / vol / pitch` 越界**同理会报错，务必在请求前夹紧：
  ```python
  speed = max(0.5, min(2.0, speed))
  vol   = max(0.0, min(10.0, vol))
  pitch = max(-12, min(12, pitch))
  ```
- **单次文本过长**会被截断；保险做法按 `~4000` 字切片循环调。
- **`audio_setting.format=wav` 不存在**，要 wav 就填 `pcm` 再把后缀改成 `.wav`，文件头需要自己补 44 字节 RIFF/WAVE header（简单做法：先拿 pcm，再 `ffmpeg -f s16le -ar 32000 -ac 1 -i raw.pcm out.wav` 转，或直接用 `pydub`）。

---

## 3. Python 最小示例（urllib，零三方依赖）

### 3.1 LLM

```python
import json, os, urllib.request

url = "https://api.MiniMax.com/v1/chat/completions"
payload = {
    "model": "MiniMax-M3",
    "messages": [{"role": "user", "content": "用一句话介绍长安。"}],
    "max_completion_tokens": 200,
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}",
        "Content-Type":  "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read().decode("utf-8"))
print(body["choices"][0]["message"]["content"])
```

### 3.2 TTS（hex → 文件）

```python
import json, os, urllib.request

url = "https://api.MiniMax.com/v1/t2a_v2"
payload = {
    "model": "speech-02-hd",
    "text": "你好，世界。<#0.8#>这是一段测试。",
    "stream": False,
    "language_boost": "Chinese",
    "voice_setting": {
        "voice_id": "male-qn-qingse",
        "speed": 1.0, "vol": 1.0, "pitch": 0,
        "emotion": "calm",
    },
    "audio_setting": {
        "sample_rate": 32000, "bitrate": 128000,
        "format": "mp3", "channel": 1,
    },
    "pronunciation_dict": {"tone": ["处理/(chu3)(li3)"]},
}
req = urllib.request.Request(
    url,
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {os.environ['MINIMAX_API_KEY']}",
        "Content-Type":  "application/json",
    },
    method="POST",
)
with urllib.request.urlopen(req, timeout=60) as resp:
    body = json.loads(resp.read().decode("utf-8"))

audio_hex = body["data"]["audio"]
length_ms = body["extra_info"]["audio_length"]   # 注意：毫秒
status    = body["base_resp"]["status_code"]
assert status == 0, body["base_resp"]

with open("out.mp3", "wb") as f:
    f.write(bytes.fromhex(audio_hex))
print(f"已写入 out.mp3，时长 {length_ms/1000:.2f}s")
```

---

## 4. 速查：字段合法范围

| 字段 | 范围 |
| --- | --- |
| `speed` | 0.5 – 2.0 |
| `vol` | 0 – 10 |
| `pitch` | -12 – 12（半音） |
| `emotion` | happy / sad / angry / fearful / disgusted / surprised / calm / fluent / whisper |
| `sample_rate` | 8000 / 16000 / 22050 / 32000 / 44100 |
| `bitrate`（mp3） | 32000 / 64000 / 128000 / 256000 |
| `audio_length` | 毫秒（ms） |
| `text` 单次 | ~10000 字符 |

---

## 5. 与本项目 drama.py 的对应

| drama.py | API |
| --- | --- |
| `analyze` 子命令 | `POST /v1/chat/completions` |
| `script.json.roles[].voice_id` | `voice_setting.voice_id` |
| `script.json.lines[].{emotion,vol,speed,pitch}` | `voice_setting.{emotion,vol,speed,pitch}` |
| `script.json.lines[].pause_after_ms` | 合成后端用静音段拼接（不是 API 字段） |
| `script.json.lines[].pronunciation` | `pronunciation_dict.tone` 或行内 `(…)` |
| `script.json.tts_model` | TTS 请求的 `model` |
| `script.json.language_boost` | TTS 请求的 `language_boost` |