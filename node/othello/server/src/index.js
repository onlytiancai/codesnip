const express = require('express');
const cors = require('cors');
const http = require('http');
const WebSocket = require('ws');
const { setupWebSocket } = require('./websocket');
const { RoomManager } = require('./game/roomManager');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// 记录所有活跃的WebSocket连接
const activeConnections = new Set();

// 配置CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST'],
}));

// 创建房间管理器
const roomManager = new RoomManager();

// 设置WebSocket服务器
setupWebSocket(wss, roomManager, activeConnections);

const PORT = process.env.PORT || 3001;
server.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});

// 导出活跃连接集合，供其他模块使用
module.exports = { activeConnections };