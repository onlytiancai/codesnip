const express = require('express');
const path = require('path');
const cors = require('cors');

const app = express();
const port = 3000;

// 配置CORS，允许任何域名访问
app.use(cors());

// 映射 /onnx-community/ 路径到 /Volumes/data/coscdn/onnx-community/ 目录
app.use('/onnx-community', express.static('/Volumes/data/coscdn/onnx-community'));

// 启动服务器
app.listen(port, () => {
  console.log(`Server is running at http://localhost:${port}`);
  console.log(`Static files mapped at http://localhost:${port}/onnx-community`);
});
