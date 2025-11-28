from arkitect.core.component.context.context import Context
from enum import Enum
import asyncio
from pydantic import Field
def get_current_weather(location: str = Field(description="地点的位置信息，例如北京、上海"), unit: str=Field(description="温度单位, 可选值为摄氏度或华氏度")):
    """
    获取指定地点的天气信息
    """
    return f"{location}今天天气晴朗，温度 25 {unit}。"
async def chat_with_tool():
    ctx = Context(
            model="doubao-1-5-pro-32k-250115",
            tools=[
                get_current_weather
            ],  # 直接在这个list里传入你的所有python 方法作为tool，tool的描述会自动带给模型推理，tool的执行在ctx.completions.create 中会自动进行
        )
    await ctx.init()
    completion = await ctx.completions.create(messages=[
        {"role": "user", "content": "北京和上海今天的天气如何？"}
    ],stream =False)
    return completion
completion = asyncio.run(chat_with_tool())
print(completion.choices[0].message.content)