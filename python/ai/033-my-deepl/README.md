设置不同模型，并打开日志

    ANTHROPIC_MODEL=MiniMax-M3.1-Flash-Preview python interactive_translate.py --debug

调试启动 server

    DEBUG=1 ANTHROPIC_API_KEY=sk-ant-... python -m uvicorn server:app --port 8000

手工测试 API

    # 翻译
    curl -X POST http://127.0.0.1:8000/api/translate \
        -H 'Content-Type: application/json' \
        -d '{"text":"埃隆·马斯克不在领英上。"}'

    # 重写
    curl -X POST http://127.0.0.1:8000/api/rephrase \
        -H 'Content-Type: application/json' \
        -d '{
            "source":"埃隆·马斯克不在领英上。",
            "current":"Elon Musk is not on LinkedIn.",
            "prefix":"Elon Musk",
            "target_word":"is"
        }'
