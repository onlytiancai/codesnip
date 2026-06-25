curl https://api.stepfun.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $STEP_API_KEY" \
  -d '{
    "model": "step-3.7-flash",
    "messages": [
      {
        "role": "system",
        "content": "你是由阶跃星辰提供的AI聊天助手，你擅长中文，英文，以及多种其他语言的对话。在保证用户数据安全的前提下，你能对用户的问题和请求，作出快速和精准的回答。同时，你的回答和建议应该拒绝黄赌毒，暴力恐怖主义的内容。"
      },
      {
        "role": "user",
        "content": "你好，请介绍一下阶跃星辰的人工智能！"
      }
    ]
  }'