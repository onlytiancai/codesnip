const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// 把 public 目录作为静态资源目录
app.use(express.static(path.join(__dirname, 'public')));

// 生成随机昵称
function randomNick() {
  return 'User-' + Math.floor(1000 + Math.random() * 9000);
}

// 简单广播（不会记录历史）
function broadcast(data, except) {
  const s = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN && client !== except) {
      client.send(s);
    }
  });
}

wss.on('connection', (ws, req) => {
  // 给连接分配昵称
  const nick = randomNick();
  ws.nick = nick;

  // 向新连接的客户端发送分配的昵称（服务端分配）
  ws.send(JSON.stringify({ type: 'assign', nick }));

  // 告知其他人有人加入（可选）
  broadcast({ type: 'join', nick, ts: Date.now() }, ws);

  // 心跳（保持连接健康）
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

  ws.on('message', raw => {
    let msg;
    try {
      msg = JSON.parse(raw);
    } catch (e) {
      return; // 忽略不可解析的数据
    }

    // 处理不同类型的消息
    // 改昵称请求（前端会发 {type: 'nick', nick: 'newNick'}）
    if (msg.type === 'nick' && typeof msg.nick === 'string') {
      const newNick = msg.nick.trim().slice(0, 32);
      if (!newNick) return;
      const oldNick = ws.nick;
      ws.nick = newNick;
      // 给请求者发回确认（assign 保持兼容）
      ws.send(JSON.stringify({ type: 'assign', nick: ws.nick }));
      // 广播昵称更改事件给其他人
      broadcast({ type: 'nick', oldNick, newNick, ts: Date.now() }, ws);
      return;
    }

    // 只处理 text 类型的发送（前端会发 {type: 'message', text: '...'}）
    if (msg.type === 'message' && typeof msg.text === 'string') {
      const out = {
        type: 'message',
        nick: ws.nick,
        text: msg.text.slice(0, 1000), // 限长
        ts: Date.now(),
      };
      broadcast(out);
    }
  });

  ws.on('close', () => {
    broadcast({ type: 'leave', nick: ws.nick, ts: Date.now() });
  });
});

// 清理死连接
const interval = setInterval(() => {
  wss.clients.forEach(ws => {
    if (!ws.isAlive) return ws.terminate();
    ws.isAlive = false;
    ws.ping(() => {});
  });
}, 30000);

server.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});

process.on('SIGTERM', () => { clearInterval(interval); server.close(); });