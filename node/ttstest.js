// https://cloud.tencent.com/document/product/1073/37995
// npm install tencentcloud-sdk-nodejs-tts --save

require('dotenv').config();
const { tts } = require("tencentcloud-sdk-nodejs-tts")
TTSClient = tts.v20190823.Client
const client = new TTSClient({
  credential: {
    secretId: process.env.TENCENTCLOUD_SECRET_ID,
    secretKey: process.env.TENCENTCLOUD_SECRET_KEY,
  }
})

const fsp = require('fs').promises;

async function textToVoice(text) {
    try {
        const data = await client.TextToVoice({
            VoiceType: 601000, // https://cloud.tencent.com/document/product/1073/92668
            EmotionCategory: 'sajiao',
            Text: text,
            SessionId: "session-1234",
            Codec: 'mp3',
            EnableSubtitle: true,
        });

        console.log('get response', data)
        const buffer = Buffer.from(data.Audio, 'base64');
        await fsp.writeFile('output.mp3', buffer);
        console.log('MP3 文件已成功保存!');
    } catch (err) {
        console.error("发生错误:", err);
    }
}

textToVoice("田脑师，你个死鬼，上哪儿浪去了，还不回家？How are you doing today?");
