# JSON 数据文件格式规范

## 文件命名
- `N.json`（N 为序号，从 1 开始递增）

## 字段说明

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `scene_en` | string | ✓ | 场景英文名 |
| `scene_zh` | string | ✓ | 场景中文名 |
| `task_en` | string | ✓ | 任务英文名 |
| `task_zh` | string | ✓ | 任务中文名 |
| `context` | string | ✓ | 上下文背景（中文描述） |
| `sentence_zh` | string | ✓ | 中文原句 |
| `translations` | array | ✓ | 英文翻译数组（3-5 条） |
| `explanation` | string | ✓ | 整体讲解 |

## translations 数组每项

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `sentence` | string | ✓ | 英文句子 |
| `style` | string | ✓ | 风格：`polite` / `neutral` / `casual` |
| `note` | string | ✓ | 简短说明 |
| `literal_translation` | string | ✓ | 中文直译 |
| `phonetic` | string | ✓ | 标准 IPA 音标（英式发音） |

## 音标规范（重点）

使用**标准 IPA（国际音标）**，英式发音（RP），重音符号用 `ˈ`，次重音用 `ˌ`。
音标两侧不需要斜杠。
句中如有明显停顿，可用 `.` 标记。

### 正确示例
```json
"phonetic": "naɪs tuː miːt juː"
"phonetic": "ɪts ə ˈpleʒə tuː miːt juː"
"phonetic": "haʊ wɜː ju dʊɪŋ"
```

### 常见错误
- ❌ 使用非标准符号：`oo` `sh` `ch` `ng`
- ❌ 使用连字符或空格分隔音节：`wʌt-ɑː-juː`
- ❌ 元音缺少长音标记：`tu` 而非 `tuː`
- ❌ 使用美式发音：`wʌt` 而非英式 `wɒt`
- ❌ 忘记 r 化音：英式句末 r 通常不发音，写 `hɪə` 而非 `hɪr`

### IPA 音标对照（常见）

| 单词 | IPA |
|------|-----|
| you | juː |
| to | tuː |
| nice | naɪs |
| meet | miːt |
| good | gʊd |
| great | greɪt |
| this | ðɪs |
| is | ɪz |
| the | ðə |
| and | ænd |
| what | wɒt |
| are | ɑː |
| for | fɔː |
| of | əv |
| it | ɪt |
| in | ɪn |
| please | pliːz |
| pleasure | ˈpleʒə |
| doing | ˈduːɪŋ |
| here | hɪə |
| there | ðeə |

## 风格分布
每条中文句子对应 3-5 条英文翻译，应包含：
- `polite`：正式礼貌（1-2 条）
- `neutral`：中性通用（1-2 条）
- `casual`：口语随意（1-2 条）

## 完整示例

```json
{
  "scene_en": "Work",
  "scene_zh": "工作",
  "task_en": "Networking",
  "task_zh": "社交",
  "context": "在行业交流会上，你遇到一位陌生人，想上前打招呼并自我介绍。",
  "sentence_zh": "你好，很高兴认识你。我是做产品设计的，你呢？",
  "translations": [
    {
      "sentence": "It's a pleasure to meet you. I'm in product design. And you?",
      "style": "polite",
      "note": "适合正式场合初次见面，友好且有礼貌",
      "literal_translation": "很荣幸见到你。我在产品设计领域。你呢？",
      "phonetic": "ɪts ə ˈpleʒə tuː miːt juː. aɪm ɪn ˈprɒdʌkt dɪˈzaɪn. ænd juː"
    }
  ],
  "explanation": "职场networking常用开场白。先表达礼貌热情，再简述职业，最后询问对方。"
}
```
