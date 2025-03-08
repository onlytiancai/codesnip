'''
大模型的 function call 测试:
提供两个函数
- 一个是根据姓名获取所在城市
- 一个是根据城市获取天气。

问 AI 某个人那里的天气怎么样，它会根据需要调用相关的 API，如果需要调用多个API它也会多次调用，最终给出一个准确的答案。

有了这个功能，可以大大扩展大模型的能力，可以让大模型做很多事情。

以下是测试输出

$ python3 ark_func_call_test.py '小明那里的天气怎么样？'
Bot [1.113 s][Use FC]:  function name=get_city_by_name, args=[{"name": "小明"}], resp=[北京]
Bot [1.973 s][Use FC]:  function name=get_weather_by_city, args=[{"city": "北京"}], resp=[晴天，气温 15-20 度。]
Bot [1.200 s][FC Summary]:  小明所在的北京天气是晴天，气温 15 - 20 度。

$ python3 ark_func_call_test.py '小红所在城市的天气如何？'
Bot [1.271 s][Use FC]:  function name=get_city_by_name, args=[{"name": "小红"}], resp=[深圳]
Bot [1.276 s][Use FC]:  function name=get_weather_by_city, args=[{"city": "深圳"}], resp=[小雨，气温 5-10 度。]
Bot [0.941 s][FC Summary]:  小红所在的深圳目前的天气是小雨，气温 5-10 度。

$ python3 ark_func_call_test.py '小王在哪个城市'
Bot [1.300 s][Use FC]:  function name=get_city_by_name, args=[{"name": "小王"}], resp=[获取数据出错]
Bot [2.191 s][FC Summary]:  很抱歉，在获取小王所在城市的数据时出现了错误，暂时无法为你提供小王所在的城市信息。你可以更换姓名再次询问或者有其他需求也可以告诉我。
'''

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
