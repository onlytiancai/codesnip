"""
MiniMax TTS 系统音色目录（voice catalog）。

音色 ID 来自 MiniMax 官方 t2a_v2 接口常见系统音色。
供 LLM 选角时按 gender/age/style 标签匹配角色，
也作为人工事后校验的参考池（不在表中的 voice_id 仍可写入，但建议人工复核）。

完整最新清单以官方文档为准：
https://platform.MiniMax.io/docs/api-reference/voice-cloning
"""

# 性别枚举
GENDER_MALE = "male"
GENDER_FEMALE = "female"

# 年龄区间
AGE_CHILD = "child"      # 儿童
AGE_YOUTH = "youth"      # 青年（18–35）
AGE_MIDDLE = "middle"    # 中年（36–55）
AGE_OLD = "old"          # 老年（55+）

# 风格标签
STYLE_NARRATION = "narration"    # 旁白/叙述
STYLE_NARRATOR_M = "narrator-male"
STYLE_NARRATOR_F = "narrator-female"
STYLE_HOST = "host"              # 主持/播报
STYLE_GENTLE = "gentle"          # 温柔
STYLE_LIVELY = "lively"          # 活泼
STYLE_MATURE = "mature"          # 成熟
STYLE_WARM = "warm"              # 温暖
STYLE_YOUNG = "young"            # 少年气
STYLE_HEROIC = "heroic"          # 英气/侠气
STYLE_ELITE = "elite"            # 精英/职业
STYLE_RURAL = "rural"            # 乡土/朴实
STYLE_OLD = "old-timber"         # 老者/沧桑
STYLE_VILLAIN = "villain"        # 霸道/反派
STYLE_AUDIOBOOK = "audiobook"    # 有声书专用

# 主流音色池（涵盖男/女、青年/中年/老年，足够武侠/言情/历史等中文有声剧）
VOICES = {
    # ---- 男声 ----
    "male-qn-qingse":    {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_YOUNG,   "desc": "青年男声，清澈"},
    "male-qn-jingying":  {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_ELITE,   "desc": "青年男声，精英"},
    "male-qn-badao":     {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_VILLAIN, "desc": "青年男声，霸道"},
    "male-qn-zhixing":   {"gender": GENDER_MALE,   "age": AGE_MIDDLE, "style": STYLE_ELITE,   "desc": "中年男声，知性"},
    "male-qn-chengshu":  {"gender": GENDER_MALE,   "age": AGE_MIDDLE, "style": STYLE_MATURE,  "desc": "中年男声，成熟"},
    "male-qn-yangguang": {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_LIVELY,  "desc": "青年男声，阳光"},
    "male-qn-wenrou":    {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_GENTLE,  "desc": "青年男声，温柔"},

    # ---- 女声 ----
    "female-shaonv":     {"gender": GENDER_FEMALE, "age": AGE_YOUTH,  "style": STYLE_LIVELY,  "desc": "少女声，活泼"},
    "female-yujie":      {"gender": GENDER_FEMALE, "age": AGE_YOUTH,  "style": STYLE_MATURE,  "desc": "御姐声，成熟"},
    "female-chengshu":   {"gender": GENDER_FEMALE, "age": AGE_MIDDLE, "style": STYLE_MATURE,  "desc": "中年女声，成熟"},
    "female-tianmei":    {"gender": GENDER_FEMALE, "age": AGE_YOUTH,  "style": STYLE_WARM,    "desc": "青年女声，甜美"},
    "female-wenrou":     {"gender": GENDER_FEMALE, "age": AGE_YOUTH,  "style": STYLE_GENTLE,  "desc": "青年女声，温柔"},

    # ---- 主持/旁白 ----
    "presenter_male":    {"gender": GENDER_MALE,   "age": AGE_MIDDLE, "style": STYLE_HOST,        "desc": "男性主持声，稳重"},
    "presenter_female":  {"gender": GENDER_FEMALE, "age": AGE_MIDDLE, "style": STYLE_HOST,        "desc": "女性主持声，清晰"},
    "audiobook_male_1":  {"gender": GENDER_MALE,   "age": AGE_MIDDLE, "style": STYLE_AUDIOBOOK,   "desc": "有声书男声 1，沉稳"},
    "audiobook_male_2":  {"gender": GENDER_MALE,   "age": AGE_YOUTH,  "style": STYLE_AUDIOBOOK,   "desc": "有声书男声 2，磁性"},
    "audiobook_female_1":{"gender": GENDER_FEMALE, "age": AGE_MIDDLE, "style": STYLE_AUDIOBOOK,   "desc": "有声书女声 1，叙述"},
    "audiobook_female_2":{"gender": GENDER_FEMALE, "age": AGE_YOUTH,  "style": STYLE_AUDIOBOOK,   "desc": "有声书女声 2，清亮"},
}


def list_voices():
    """返回供 LLM prompt 注入的音色清单字符串。"""
    lines = []
    for vid, meta in VOICES.items():
        lines.append(
            f"- {vid}: {meta['desc']}（{meta['gender']}/{meta['age']}/{meta['style']}）"
        )
    return "\n".join(lines)


def lookup(voice_id):
    """查表；找不到返回 None（不抛异常，便于 LLM 任意给 ID 后回退）。"""
    return VOICES.get(voice_id)


def gender_of(voice_id):
    """获取 voice_id 对应性别；未知则返回 None。"""
    info = lookup(voice_id)
    return info["gender"] if info else None