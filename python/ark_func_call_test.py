import config
import json
import sys
from volcenginesdkarkruntime import Ark
import time
from pprint import pprint

prompt = sys.argv[1]

client = Ark(api_key=config.ARK_API_KEY)

city_map = {
    '小明': '北京',
    '小红': '深圳',
}

weather_map = {
    '北京': '晴天，气温 15-20 度。',
    '深圳': '小雨，气温 5-10 度。',
}

tool_city = {
    "type": "function",
    "function": {
        "name": "get_city_by_name",
        "description": "根据姓名获取所在城市",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "姓名，如小王，小明"},
             },
            "required": ["name"],
        },
    },
}
tool_weather = {
    "type": "function",
    "function": {
        "name": "get_weather_by_city",
        "description": "根据城市获取天气",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名，比如北京，天津"},
             },
            "required": ["city"],
        },
    },
}


messages = [
    {
        "role": "system",
        "content": "你是蛙蛙，一个 AI 人工智能助手",
    },
    {
        "role": "user",
        "content": prompt,
    },
]

req = {
    "model": "doubao-1-5-pro-32k-250115",
    "messages": messages,
    "temperature": 0.8,
    "tools": [tool_weather, tool_city],
}

req_times = 0
while True:
    req_times += 1
    if req_times > 5:
        print('请求次数太多')
        break
    ts = time.time()
    completion = client.chat.completions.create(**req)
    # pprint(completion.model_dump())
    if completion.choices[0].message.tool_calls:
        tool_call = completion.choices[0].message.tool_calls[0]
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        func_resp = '获取数据出错' 
        if func_name == 'get_weather_by_city':
            func_resp = weather_map.get(func_args['city'], '获取数据出错')
        elif func_name == 'get_city_by_name':
            func_resp = city_map.get(func_args['name'], '获取数据出错')
        else:
            raise Exception(f'unknow {tool_call.function.name}')

        print(
            f"Bot [{time.time() - ts:.3f} s][Use FC]: ",
            f'function name={func_name}, args=[{json.dumps(func_args, ensure_ascii=False)}], resp=[{func_resp}]'
        )
        req["messages"].extend(
            [
                completion.choices[0].message.model_dump(),
                 {
                    "role": "tool",
                    "tool_call_id": completion.choices[0].message.tool_calls[0].id,
                    "content": func_resp,
                    "name": completion.choices[0].message.tool_calls[0].function.name,
                },
            ]
        )

        # pprint(req['messages'])
    else:
        print(
            f"Bot [{time.time() - ts:.3f} s][FC Summary]: ",
            completion.choices[0].message.content,
        )
        break
