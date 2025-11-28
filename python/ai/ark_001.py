import os
from volcenginesdkarkruntime import Ark

client = Ark(
    api_key=os.environ.get("ARK_API_KEY"),
    # The base URL for model invocation .
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    )
completion = client.chat.completions.create(
    # Replace with Model ID .
    model="doubao-seed-1-6-251015",
    messages=[
        {"role": "user", "content": "Hello"}
    ]
)
print(completion.choices[0].message)