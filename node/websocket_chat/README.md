这个小项目包含一个 **Node.js WebSocket 服务端** 和一个 **前端 HTML 客户端**。

**特点**
- 服务端不保存任何聊天历史；新加入的人只能看到之后的消息。
- 每个连接会被分配一个随机昵称（例如 `User-4921`），昵称由服务端下发。
- 使用简单的 JSON 协议（`type` 字段）传递消息。
- 支持加入/离开广播，但不会回放历史。

## 运行说明（README）

1. 确保你已安装 Node.js（>=14）
2. 在项目根目录运行：

```bash
npm init -y
npm install express ws
node server.js
```

3. 打开浏览器访问 `http://localhost:3000`，打开多个标签页或不同设备测试实时聊天。

**注意与扩展**
- 目前示例使用 `ws://`（非加密），部署到公网时请使用 HTTPS + WSS（TLS）。
- 服务端 **不保存消息历史**，符合你的要求：新加入的客户端只会接收到之后发生的 join/message/leave 事件。
- 可以把 `broadcast({ type: 'join', nick, ts: ... })` 改为不广播以隐藏加入/离开通知。

---

如果你需要，我可以：
- 把这个项目打包成一个可下载的 zip；
- 改用 `uWebSockets.js` 或 `socket.io`；
- 增加用户名自定义、房间（channel）支持、或者把消息记录到 Redis（但那会违背“服务端不保存历史”的要求）。
