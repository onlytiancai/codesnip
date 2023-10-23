curl --location --request POST "https://${SPEECH_REGION}.tts.speech.microsoft.com/cognitiveservices/v1" \
--header "Ocp-Apim-Subscription-Key: ${SPEECH_KEY}" \
--header 'Content-Type: application/ssml+xml' \
--header 'X-Microsoft-OutputFormat: audio-16khz-128kbitrate-mono-mp3' \
--header 'User-Agent: curl' \
--data-raw '<speak version='\''1.0'\'' xml:lang='\''en-US'\''>
    <voice xml:lang='\''zh-CN'\'' xml:gender='\''Female'\'' name='\''zh-CN-XiaoxiaoNeural'\''>
每个人的心中都有一座花园
你若荒芜，则寸草不生
你若热爱，则繁花盛开
生命本没有意义
世事无法尽如人意
努力也不一定有结果
我们能做的就是活在当下
去看自己想看的风景
去过自己热爱的生活
    </voice>
</speak>' > output.mp3
