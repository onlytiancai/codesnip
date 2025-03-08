import config
from volcenginesdkarkruntime import Ark
import time
client = Ark(api_key=config.ARK_API_KEY)

def test_function_call():
    messages = [
        {
            "role": "system",
            "content": "你是豆包，是由字节跳动开发的 AI 人工智能助手",
        },
        {
            "role": "user",
            "content": "北京今天的天气",
        },
    ]
    req = {
        "model": "doubao-1-5-pro-32k-250115",
        "messages": messages,
        "temperature": 0.8,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "MusicPlayer",
                    "description": """歌曲查询Plugin，当用户需要搜索某个歌手或者歌曲时使用此plugin，给定歌手，歌名等特征返回相关音乐。\n 例子1：query=想听孙燕姿的遇见， 输出{"artist":"孙燕姿","song_name":"遇见","description":""}""",
                    "parameters": {
                        "properties": {
                            "artist": {"description": "表示歌手名字", "type": "string"},
                            "description": {
                                "description": "表示描述信息",
                                "type": "string",
                            },
                            "song_name": {
                                "description": "表示歌曲名字",
                                "type": "string",
                            },
                        },
                        "required": [],
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "地理位置，比如北京市",
                            },
                            "unit": {"type": "string", "description": "枚举值 [摄氏度,华氏度]"},
                        },
                        "required": ["location"],
                    },
                },
            },
        ],
    }

    ts = time.time()
    completion = client.chat.completions.create(**req)
    if completion.choices[0].message.tool_calls:
        print(
            f"Bot [{time.time() - ts:.3f} s][Use FC]: ",
            completion.choices[0].message.tool_calls[0],
        )
        # ========== 补充函数调用的结果 =========
        req["messages"].extend(
            [
                completion.choices[0].message.dict(),
                 {
                    "role": "tool",
                    "tool_call_id": completion.choices[0].message.tool_calls[0].id,
                    "content": "北京天气多云，-5到3度",  # 根据实际调用函数结果填写，最好用自然语言。
                    "name": completion.choices[0].message.tool_calls[0].function.name,
                },
            ]
        )
        # 再请求一次模型，获得总结。 如不需要，也可以省略
        ts = time.time()
        completion = client.chat.completions.create(**req)
        print(
            f"Bot [{time.time() - ts:.3f} s][FC Summary]: ",
            completion.choices[0].message.content,
        )

test_function_call()
