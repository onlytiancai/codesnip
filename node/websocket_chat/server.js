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
  let sent = 0;
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN && client !== except) {
      try {
        client.send(s);
        sent++;
      } catch (err) {
        // 不阻塞广播流程
        console.error('Broadcast send error to client:', err && err.message ? err.message : err);
      }
    }
  });
  // 简短日志，避免太多输出
  if (data && data.type) {
    console.log(`[broadcast] type=${data.type} sent=${sent}` + (data.nick ? ` from=${data.nick}` : ''));
  } else {
    console.log(`[broadcast] sent=${sent}`);
  }
}

wss.on('connection', (ws, req) => {
  // 给连接分配昵称
  const nick = randomNick();
  ws.nick = nick;
  // 提取客户端 IP 并做简单掩码处理（只显示第一段和最后一段）
  function maskIp(ip) {
    if (!ip) return '';
    // IPv6 链接本地或带端口的 IPv4 地址可能出现，例如 ::1、::ffff:192.168.1.80
    // 先剥离可能的前缀和端口
    let raw = ip;
    // 如果是 ws._socket.remoteAddress 风格，会直接是 IP
    // 如果带有 IPv6 前缀 ::ffff: 的 IPv4 地址，转换为纯 IPv4
    if (raw.startsWith('::ffff:')) raw = raw.replace('::ffff:', '');
    // 去掉可能的端口（不太可能在 remoteAddress 中出现，但保险处理）
    if (raw.includes('%')) raw = raw.split('%')[0];
    if (raw.includes(':') && raw.indexOf(':') === raw.lastIndexOf(':')) {
      // 单个冒号可能是 IPv6 里带端口的表示，尽量剥离端口
      const parts = raw.split(':');
      if (parts.length > 1 && parts[parts.length - 1].match(/^\d+$/)) raw = parts.slice(0, -1).join(':');
    }

    const parts = raw.split('.');
    if (parts.length === 4) {
      return `${parts[0]}.*.*.${parts[3]}`;
    }
    // 对 IPv6 或其他格式，简单掩码中间部分
    const segs = raw.split(':');
    if (segs.length >= 2) {
      return `${segs[0]}:*:${segs[segs.length - 1]}`;
    }
    return raw;
  }

  const remoteAddr = (req && (req.socket && req.socket.remoteAddress)) || (ws && ws._socket && ws._socket.remoteAddress) || '';
  ws.ip = maskIp(remoteAddr);
  // 保留原始地址用于 server 日志显示（不对外广播原始地址）
  ws.remoteAddr = remoteAddr;

  console.log(`[connect] nick=${ws.nick} ip=${ws.remoteAddr}`);

  // 向新连接的客户端发送分配的昵称（服务端分配）
  ws.send(JSON.stringify({ type: 'assign', nick, ip: ws.ip }));

  // 告知其他人有人加入（可选）
  broadcast({ type: 'join', nick, ip: ws.ip, ts: Date.now() }, ws);

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
  ws.send(JSON.stringify({ type: 'assign', nick: ws.nick, ip: ws.ip }));
  // 广播昵称更改事件给其他人
  broadcast({ type: 'nick', oldNick, newNick, ip: ws.ip, ts: Date.now() }, ws);
  console.log(`[nick] ${oldNick} -> ${newNick} ip=${ws.remoteAddr}`);
      return;
    }

    // 只处理 text 类型的发送（前端会发 {type: 'message', text: '...'}）
    if (msg.type === 'message' && typeof msg.text === 'string') {
      const out = {
        type: 'message',
        nick: ws.nick,
        ip: ws.ip,
        text: msg.text.slice(0, 1000), // 限长
        ts: Date.now(),
      };
  broadcast(out);
  console.log(`[message] from=${ws.nick} ip=${ws.remoteAddr} textLen=${String(out.text).length} [${String(out.text)}]`);
    }

    // 处理语音消息（前端发送 data URL）
    if (msg.type === 'audio' && typeof msg.data === 'string') {
      // 不在服务器端解码，只做简单大小限制（防止被滥用）
      const maxSize = 2 * 1024 * 1024; // 2MB
      // 估算 base64 大小：data:audio/...;base64,xxxx
      const base64Part = msg.data.split(',')[1] || '';
      const estimatedBytes = Math.ceil((base64Part.length * 3) / 4);
      if (estimatedBytes > maxSize) return; // 忽略过大的语音
      const out = {
        type: 'audio',
        nick: ws.nick,
        ip: ws.ip,
        data: msg.data,
        ts: Date.now(),
      };
      broadcast(out);
  console.log(`[audio] from=${ws.nick} ip=${ws.remoteAddr} size=${estimatedBytes}B`);
    }
  });

  ws.on('close', () => {
  broadcast({ type: 'leave', nick: ws.nick, ip: ws.ip, ts: Date.now() });
  console.log(`[close] nick=${ws.nick} ip=${ws.remoteAddr}`);
  });
  ws.on('error', err => { console.error('[ws error]', err && err.message ? err.message : err); });
});

// 清理死连接
const interval = setInterval(() => {
  wss.clients.forEach(ws => {
    if (!ws.isAlive) return ws.terminate();
    ws.isAlive = false;
    ws.ping(() => {});
  });
}, 30000);

server.listen(3005, () => {
  console.log('Server running on http://localhost:3005');
});

process.on('SIGTERM', () => { clearInterval(interval); server.close(); });