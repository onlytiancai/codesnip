from ollama import chat

response = chat(
    model='s2021008840/hy-mt2:1.8b',
    messages=[{'role': 'user', 'content': 'Hello!'}],
)
print(response.message.content)
