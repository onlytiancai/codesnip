const express = require('express');
const axios = require('axios');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = 3000;

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// VolcEngine TTS API Configuration
const VOLT_TTS_CONFIG = {
  endpoint: process.env.VOLT_ENDPOINT || 'https://openspeech.bytedance.com/api/v3/tts/unidirectional',
  appid: process.env.VOLT_APPID,
  accessKey: process.env.VOLT_ACCESS_KEY,
  secretKey: process.env.VOLT_SECRET_KEY,
  resourceID: process.env.VOLT_RESOURCE_ID || 'seed-tts-1.0'
};

// TTS API endpoint with streaming support using native https module
app.post('/api/tts', async (req, res) => {
  const requestId = Date.now().toString();
  console.log(`[${requestId}] ===== TTS 请求开始 =====`);
  
  try {
    const { text, voiceType, speedRatio, volumeRatio } = req.body;
    
    console.log(`[${requestId}] 请求参数:`, {
      textLength: text ? text.length : 0,
      textPreview: text ? text.substring(0, 50) + (text.length > 50 ? '...' : '') : '',
      voiceType,
      speedRatio,
      volumeRatio
    });
    
    if (!text) {
      console.log(`[${requestId}] 错误: 文本为空`);
      return res.status(400).json({ error: 'Text is required' });
    }

    // Prepare request parameters
    const params = {
      "user": {
        "uid": "12345"
      },
      "req_params": {
        "text": text,
        "speaker": voiceType || "zh_female_cancan_mars_bigtts",
        "audio_params": {
          "format": "mp3",
          "sample_rate": 24000,
          "enable_timestamp": true
        },
        "additions": "{\"explicit_language\":\"zh\",\"disable_markdown_filter\":true, \"enable_timestamp\":true}\""
      }
    };

    console.log(`[${requestId}] 请求火山引擎 API: ${VOLT_TTS_CONFIG.endpoint}`);

    // Use axios but with better stream handling
    const response = await axios({
      method: 'post',
      url: VOLT_TTS_CONFIG.endpoint,
      headers: {
        'Content-Type': 'application/json',
        'X-Api-App-Id': VOLT_TTS_CONFIG.appid,
        'X-Api-Access-Key': VOLT_TTS_CONFIG.accessKey,
        'X-Api-Resource-Id': VOLT_TTS_CONFIG.resourceID,
        'Connection': 'keep-alive',
        'Accept': 'text/event-stream'
      },
      data: params,
      responseType: 'stream',
      timeout: 60000,
      maxContentLength: Infinity,
      maxBodyLength: Infinity
    });

    console.log(`[${requestId}] 火山引擎 API 响应状态: ${response.status}`);
    const logid = response.headers['x-tt-logid'];
    if (logid) {
      console.log(`[${requestId}] X-Tt-Logid: ${logid}`);
    }

    // Process stream
    let audioChunks = [];
    let totalAudioSize = 0;
    let sentenceData = null;
    let usageData = null;
    let lineBuffer = '';

    return new Promise((resolve, reject) => {
      const stream = response.data;

      stream.on('data', (chunk) => {
        console.log(`[${requestId}] 收到原始数据块，长度:`, chunk.length);
        console.log(`[${requestId}] 数据块内容（前200字符）:`, chunk.toString().substring(0, 200));
        
        lineBuffer += chunk.toString();
        
        // Split by newlines
        const lines = lineBuffer.split('\n');
        lineBuffer = lines.pop(); // Keep last incomplete line
        
        console.log(`[${requestId}] 解析到 ${lines.length} 行完整数据`);

        for (const line of lines) {
          if (!line.trim()) {
            console.log(`[${requestId}] 跳过空行`);
            continue;
          }

          console.log(`[${requestId}] 处理行数据（前150字符）:`, line.substring(0, 150));
          
          try {
            const data = JSON.parse(line);
            console.log(`[${requestId}] 解析成功: code=${data.code}, hasData=${!!data.data}, dataLength=${data.data ? data.data.length : 0}, hasSentence=${!!data.sentence}`);
            
            // Handle audio data chunks
            if (data.code === 0 && data.data) {
              // data.data is base64 encoded audio
              const audioBuffer = Buffer.from(data.data, 'base64');
              audioChunks.push(audioBuffer);
              totalAudioSize += audioBuffer.length;
              console.log(`[${requestId}] 收到音频块: ${audioBuffer.length} bytes, 总计: ${totalAudioSize} bytes`);
              continue;
            }

            // Handle sentence data
            if (data.code === 0 && data.sentence) {
              sentenceData = data;
              console.log(`[${requestId}] 句子数据:`, JSON.stringify(data.sentence));
              continue;
            }

            // Handle completion (code = 20000000)
            if (data.code === 20000000) {
              if (data.usage) {
                usageData = data.usage;
                console.log(`[${requestId}] 用量数据:`, JSON.stringify(data.usage));
              }
              console.log(`[${requestId}] 流式传输完成，共收到 ${audioChunks.length} 个音频块`);
              
              // Combine all audio chunks
              if (audioChunks.length === 0) {
                console.error(`[${requestId}] 警告: 没有收到任何音频数据`);
              }
              
              const completeAudio = Buffer.concat(audioChunks);
              console.log(`[${requestId}] 完整音频大小: ${completeAudio.length} bytes (${(completeAudio.length / 1024).toFixed(2)} KB)`);
              
              // Send response
              res.json({
                code: 0,
                message: "",
                data: completeAudio.toString('base64')
              });
              
              resolve();
              return;
            }

            // Handle errors
            if (data.code > 0 && data.code !== 20000000) {
              console.error(`[${requestId}] API 错误响应:`, JSON.stringify(data));
              res.status(500).json({
                code: data.code,
                message: data.message || 'TTS generation failed',
                data: null
              });
              resolve();
              return;
            }
          } catch (error) {
            console.error(`[${requestId}] 解析 JSON 失败:`, error.message);
            console.error(`[${requestId}] 失败的行内容:`, line);
          }
        }
      });

      stream.on('end', () => {
        console.log(`[${requestId}] ===== 流结束 =====`);
        console.log(`[${requestId}] 剩余 buffer 内容:`, lineBuffer);
        
        // If stream ended without completion signal, return what we have
        if (audioChunks.length > 0) {
          const completeAudio = Buffer.concat(audioChunks);
          console.log(`[${requestId}] 流提前结束，返回已收集的音频: ${completeAudio.length} bytes`);
          res.json({
            code: 0,
            message: "",
            data: completeAudio.toString('base64')
          });
        } else {
          console.error(`[${requestId}] 流结束但没有音频数据`);
          res.status(500).json({
            code: 1,
            message: 'Stream ended without audio data',
            data: null
          });
        }
        
        resolve();
      });

      stream.on('error', (error) => {
        console.error(`[${requestId}] 流错误:`, error.message);
        console.error(`[${requestId}] 错误堆栈:`, error.stack);
        reject(error);
      });

      stream.on('close', () => {
        console.log(`[${requestId}] ===== 流关闭 =====`);
      });
    });

  } catch (error) {
    console.error(`[${requestId}] TTS 错误:`, error.message);
    
    if (error.response) {
      console.error(`[${requestId}] 错误响应状态:`, error.response.status);
      console.error(`[${requestId}] 错误响应数据:`, error.response.data ? error.response.data.toString() : 'N/A');
    } else if (error.request) {
      console.error(`[${requestId}] 请求已发送但未收到响应`);
    } else {
      console.error(`[${requestId}] 请求配置错误:`, error.message);
    }
    
    console.error(`[${requestId}] 完整错误堆栈:`, error.stack);
    
    // Return error in the specified format
    res.status(500).json({ 
      code: 1,
      message: 'Failed to generate speech: ' + error.message,
      data: ""
    });
  }
});

// Get available voices endpoint
app.get('/api/voices', (req, res) => {
  const voices = [
    { id: 'zh_female_tianmei_moon_bigtts', name: '天美（女声）', lang: 'zh' },
    { id: 'zh_male_chunxia_moon_bigtts', name: '春晓（男声）', lang: 'zh' },
    { id: 'zh_female_doudou_moon_bigtts', name: '豆豆（女声）', lang: 'zh' },
    { id: 'zh_male_yezi_moon_bigtts', name: '叶子（男声）', lang: 'zh' },
    { id: 'zh_female_cancan_mars_bigtts', name: '灿灿（女声）', lang: 'zh' },
    { id: 'zh_female_shuangkuaisisi_moon_bigtts', name: '双快思思（女声）', lang: 'zh' },
    { id: 'en_female_amy_001', name: 'Amy（女声）', lang: 'en' },
    { id: 'en_male_jerry_001', name: 'Jerry（男声）', lang: 'en' }
  ];
  res.json(voices);
});

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', message: 'VolcEngine TTS server is running' });
});

app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
  console.log(`Environment variables loaded from .env file`);
});
