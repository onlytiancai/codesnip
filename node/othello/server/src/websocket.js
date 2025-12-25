const setupWebSocket = (wss, roomManager) => {
  wss.on('connection', (ws) => {
    console.log('New client connected');

    ws.on('message', (message) => {
      try {
        const parsedMessage = JSON.parse(message);
        handleMessage(ws, parsedMessage, roomManager);
      } catch (error) {
        console.error('Error parsing message:', error);
        ws.send(JSON.stringify({
          type: 'ERROR',
          payload: { message: 'Invalid message format' }
        }));
      }
    });

    ws.on('close', () => {
      console.log('Client disconnected');
      // 清理断开连接的客户端
      roomManager.removePlayerBySocket(ws);
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
    });
  });
};

// 处理发送聊天消息
const handleSendChat = (ws, payload, roomManager) => {
  const { roomId, message } = payload;
  const room = roomManager.getRoom(roomId);
  
  if (room) {
    const player = room.getPlayerBySocket(ws);
    if (player) {
      // 更新房间最后活动时间
      room.lastActivityTime = Date.now();
      
      // 广播聊天消息给房间内所有玩家
      roomManager.broadcastToRoom(roomId, JSON.stringify({
        type: 'CHAT_MESSAGE',
        payload: {
          playerName: player.name,
          playerColor: player.isSpectator ? null : player.color,
          message: message,
          timestamp: Date.now()
        }
      }));
    }
  }
};

const handleMessage = (ws, message, roomManager) => {
  const { type, payload } = message;

  switch (type) {
    case 'CREATE_ROOM':
      handleCreateRoom(ws, payload, roomManager);
      break;
    case 'JOIN_ROOM':
      handleJoinRoom(ws, payload, roomManager);
      break;
    case 'LEAVE_ROOM':
      handleLeaveRoom(ws, payload, roomManager);
      break;
    case 'MAKE_MOVE':
      handleMakeMove(ws, payload, roomManager);
      break;
    case 'GET_ROOM_INFO':
      handleGetRoomInfo(ws, payload, roomManager);
      break;
    case 'GET_STATS':
      handleGetStats(ws, roomManager);
      break;
    case 'SEND_CHAT':
      handleSendChat(ws, payload, roomManager);
      break;
    default:
      ws.send(JSON.stringify({
        type: 'ERROR',
        payload: { message: 'Unknown message type' }
      }));
  }
};

const handleCreateRoom = (ws, payload, roomManager) => {
  const room = roomManager.createRoom();
  console.log(`Room ${room.id} created`);
  ws.send(JSON.stringify({
    type: 'ROOM_CREATED',
    payload: { roomId: room.id }
  }));
};

const handleJoinRoom = (ws, payload, roomManager) => {
  const { roomId, playerName } = payload;
  console.log(`Player ${playerName} trying to join room ${roomId}`);
  const result = roomManager.addPlayerToRoom(roomId, ws, playerName);
  
  if (result.success) {
    console.log(`Player ${playerName} joined room ${roomId} successfully`);
    ws.send(JSON.stringify({
      type: 'JOINED_ROOM',
      payload: {
        roomId: roomId,
        playerId: result.player.id,
        playerColor: result.player.color,
        room: result.room
      }
    }));
    // 广播玩家加入通知给房间内所有玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'PLAYER_JOINED',
      payload: {
        playerName: playerName,
        playerColor: result.player.color,
        isSpectator: result.player.isSpectator
      }
    }));
    // 广播房间更新给其他玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'ROOM_UPDATED',
      payload: { room: result.room }
    }), [ws]);
  } else {
    console.log(`Player ${playerName} failed to join room ${roomId}: ${result.message}`);
    ws.send(JSON.stringify({
      type: 'ERROR',
      payload: { message: result.message }
    }));
  }
};

const handleLeaveRoom = (ws, payload, roomManager) => {
  const { roomId } = payload;
  roomManager.removePlayerFromRoom(roomId, ws);
};

const handleMakeMove = (ws, payload, roomManager) => {
  const { roomId, row, col } = payload;
  console.log(`Player making move in room ${roomId} at (${row}, ${col})`);
  const result = roomManager.makeMove(roomId, ws, row, col);
  
  if (result.success) {
    console.log(`Move successful in room ${roomId} at (${row}, ${col})`);
    // 广播移动结果给所有玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'MOVE_MADE',
      payload: result
    }));
    // 广播房间更新，包含最新的游戏状态和有效落子
    const room = roomManager.getRoom(roomId);
    if (room) {
      roomManager.broadcastToRoom(roomId, JSON.stringify({
        type: 'ROOM_UPDATED',
        payload: { room: room.getPublicInfo() }
      }));
    }
  } else {
    console.log(`Move failed in room ${roomId} at (${row}, ${col}): ${result.message}`);
    ws.send(JSON.stringify({
      type: 'ERROR',
      payload: { message: result.message }
    }));
  }
};

const handleGetRoomInfo = (ws, payload, roomManager) => {
  const { roomId } = payload;
  const room = roomManager.getRoom(roomId);
  
  if (room) {
    ws.send(JSON.stringify({
      type: 'ROOM_INFO',
      payload: { room }
    }));
  } else {
    ws.send(JSON.stringify({
      type: 'ERROR',
      payload: { message: 'Room not found' }
    }));
  }
};

// 处理获取统计信息请求
const handleGetStats = (ws, roomManager) => {
  const stats = {
    roomCount: roomManager.getRoomCount(),
    onlinePlayerCount: roomManager.getOnlinePlayerCount()
  };
  
  ws.send(JSON.stringify({
    type: 'STATS',
    payload: stats
  }));
};

module.exports = { setupWebSocket };