import config
import json
import sys
from volcenginesdkarkruntime import Ark
import time
from pprint import pprint

client = Ark(api_key=config.ARK_API_KEY)

print("----- standard request -----")

system_prompt = '''你是一位英语专家，负责为用户提供的英语句子进行语法分析。
假设用户提供的英语句子是：The man who lives next door is my teacher.

将英语句子翻译成中文，如：
### 中文翻译
住在隔壁的人是我的老师。


解释句子中的重点单词，如：
### 单词列表
- live: v.  /lɪv; laɪv/ 生存，活着；生活在；
- teacher: n. /ˈtiːtʃər/ 教师
</单词列表>

用多层括号和缩进在代码块内清晰表示出从句和修饰关系，如果从句是长句子要把从句的句子成分也展现出来。例如：
### 句子成分
```
The man
    (who lives next door)
is my teacher.
```

详细介绍句子中的每个成分，包括主语、谓语、宾语、定语、状语、补语等，说明它们在句子中的作用和语法功能, 如：

### 句子成分介绍
- **The man**：在主句中作主语，表示句子所描述动作或状态的执行者或主体，这里指特定的那个男人。
- **who lives next door**：是一个定语从句，修饰先行词“the man”。
    - 其中“who”是关系代词，在定语从句中作主语，用来引导定语从句并指代先行词“the man”；
    - “lives”是定语从句中的谓语动词，表示“居住”的动作；
    - “next door”是地点状语，说明居住的地点，它在定语从句中修饰“lives”，表明居住的位置是隔壁。
- **is**：是主句中的谓语动词，属于系动词，起到连接主语和表语的作用，构成主系表结构。
- **my teacher**：在主句中作表语，用于说明主语“the man”的身份，即他是“我的老师”。

请确保分析准确、全面，对句子成分的介绍清晰易懂。
如果用户询问英语语法无关的问题，拒绝回答。
如果用户发送的文字不是英语，请提示用户重新输入。'''

cmd = sys.argv[1]
if cmd == 'create_context':
    response = client.context.create(
        model= "ep-20250417200154-wqfzm", 
        messages = [ 
            {"role":"system","content": system_prompt}
        ], 
        mode= "common_prefix",
        ttl= 3600
    )
    pprint(response)
elif cmd == 'fanyi':
    context_id = sys.argv[2]
    user_prompt = sys.argv[3]
    completion = client.context.completions.create(
        context_id=context_id,
        model= "ep-20250417200154-wqfzm", 
        messages=[
            {"role": "user", "content": user_prompt},
        ],
    )
    print(completion.choices[0].message.content)
elif cmd == 'runall':
    import csv
    import time
    reader = csv.reader(open('/home/ubuntu/temp/output.csv'))
    context_id = 'ctx-20250417200431-dqhdv'
    for i, row in enumerate(reader):
        if i <= 911:
            continue
        user_prompt = row[-1]
        completion = client.context.completions.create(
            context_id=context_id,
            model= "ep-20250417200154-wqfzm", 
            messages=[
                {"role": "user", "content": user_prompt},
            ],
        )
        output = completion.choices[0].message.content
        print('progress', i, len(output))
        open(f'./fanyi_result/{i}.md', 'w').write(output)
        time.sleep(1)
