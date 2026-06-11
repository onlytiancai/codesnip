import express from 'express';
import speakeasy from 'speakeasy';

const app = express();
const port = 3001;

app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// CORS中间件
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
  
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  
  next();
});

// 生成2FA密钥
app.post('/api/2fa/generate', (req, res) => {
  try {
    const { name = 'Test 2FA Account' } = req.body;
    
    const secret = speakeasy.generateSecret({
      length: 20,
      name
    });
    
    // 手动构建包含正确issuer的otpauth URL
    const issuer = '2FA Client';
    const otpauthUrl = `otpauth://totp/${encodeURIComponent(name)}?secret=${secret.base32}&issuer=${encodeURIComponent(issuer)}`;
    
    res.json({
      success: true,
      data: {
        secret: secret.base32,
        otpauthUrl
      }
    });
  } catch (error) {
    console.error('生成密钥失败:', error);
    res.status(500).json({
      success: false,
      message: '生成密钥失败'
    });
  }
});

// 验证OTP
app.post('/api/2fa/verify', (req, res) => {
  try {
    const { secret, token } = req.body;
    
    if (!secret || !token) {
      return res.status(400).json({
        success: false,
        message: '缺少必要参数'
      });
    }
    
    const isValid = speakeasy.totp.verify({
      secret,
      encoding: 'base32',
      token
    });
    
    res.json({
      success: true,
      data: {
        valid: isValid
      }
    });
  } catch (error) {
    console.error('验证失败:', error);
    res.status(500).json({
      success: false,
      message: '验证失败'
    });
  }
});

app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
