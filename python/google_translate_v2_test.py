from google.cloud import translate_v2 as translate

# 初始化客户端（需设置环境变量 GOOGLE_APPLICATION_CREDENTIALS 指向 JSON 密钥文件）
translate_client = translate.Client()

text = "你好，世界！"
target = "en"

result = translate_client.translate(text, target_language=target)

print("原文:", text)
print("译文:", result["translatedText"])
