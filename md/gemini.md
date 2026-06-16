测试
```
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent" \
-H 'Content-Type: application/json' \
-H "X-goog-api-key: $GEMINI_API_KEY" \
-X POST \
-d '{
    "contents": [
        {
        "parts": [
            {
            "text": "Explain how AI works in a few words"
            }
        ]
        }
    ]
}'
```