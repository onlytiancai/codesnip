const setupWebSocket = (wss, roomManager, activeConnections) => {
  wss.on('connection', (ws) => {
    console.log('New client connected');
    
    // 记录新的活跃连接
    activeConnections.add(ws);

    ws.on('message', (message) => {
      try {
        const parsedMessage = JSON.parse(message);
        handleMessage(ws, parsedMessage, roomManager, activeConnections);
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
      
      // 从活跃连接集合中移除
      activeConnections.delete(ws);
      
      // 清理断开连接的客户端
      roomManager.removePlayerBySocket(ws);
    });

    ws.on('error', (error) => {
      console.error('WebSocket error:', error);
      
      // 发生错误时也移除连接
      activeConnections.delete(ws);
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

// 处理玩家重连
const handleReconnectRoom = (ws, payload, roomManager) => {
  const { roomId, playerName, playerColor } = payload;
  console.log(`Player ${playerName} trying to reconnect to room ${roomId} with color ${playerColor}`);
  const room = roomManager.getRoom(roomId);
  
  if (!room) {
    ws.send(JSON.stringify({
      type: 'ERROR',
      payload: { message: 'Room not found' }
    }));
    return;
  }
  
  // 检查该颜色是否已经有玩家在房间中
  const existingPlayer = room.players.find(p => p.color === playerColor && !p.isSpectator);
  if (existingPlayer) {
    // 如果玩家仍然在房间中，更新其socket和名字（可能用户在掉线期间改了名字）
    existingPlayer.socket = ws;
    existingPlayer.name = playerName;
    // 更新房间最后活动时间
    room.lastActivityTime = Date.now();
    
    console.log(`Player ${playerName} reconnected to room ${roomId} successfully`);
    ws.send(JSON.stringify({
      type: 'JOINED_ROOM',
      payload: {
        roomId: roomId,
        playerId: existingPlayer.id,
        playerColor: existingPlayer.color,
        room: room.getPublicInfo()
      }
    }));
    
    // 广播玩家重新连接通知给房间内其他玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'PLAYER_JOINED',
      payload: {
        playerName: playerName,
        playerColor: playerColor,
        isSpectator: false,
        isReconnect: true // 标记这是重连
      }
    }), [ws]); // 不给自己发送重连通知
  } else {
    // 如果玩家已经离开房间，作为新玩家加入
    const result = room.addPlayer(ws, playerName);
    if (result.success) {
      console.log(`Player ${playerName} reconnected to room ${roomId} as new player`);
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
    } else {
      ws.send(JSON.stringify({
        type: 'ERROR',
        payload: { message: result.message }
      }));
    }
  }
};

const handleMessage = (ws, message, roomManager, activeConnections) => {
  const { type, payload } = message;
  console.log(`Received message type: ${type}`);

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
      handleGetStats(ws, roomManager, activeConnections);
      break;
    case 'SEND_CHAT':
      handleSendChat(ws, payload, roomManager);
      break;
    case 'RESTART_GAME':
      handleRestartGame(ws, payload, roomManager);
      break;
    case 'HEARTBEAT':
      handleHeartbeat(ws, payload, roomManager);
      break;
    case 'RECONNECT_ROOM':
      handleReconnectRoom(ws, payload, roomManager);
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
    // 更新房间活动时间，防止在用户输入昵称期间被清理
    room.updateLastActivityTime();
    
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
const handleGetStats = (ws, roomManager, activeConnections) => {
  const stats = {
    roomCount: roomManager.getRoomCount(),
    onlinePlayerCount: activeConnections.size // 使用活跃连接数，而不是房间内玩家数
  };
  
  ws.send(JSON.stringify({
    type: 'STATS',
    payload: stats
  }));
};

// 处理游戏重置请求
const handleRestartGame = (ws, payload, roomManager) => {
  const { roomId } = payload;
  console.log(`Restarting game in room ${roomId}`);
  const result = roomManager.resetGame(roomId);
  
  if (!result) {
    ws.send(JSON.stringify({
      type: 'ERROR',
      payload: { message: 'Failed to reset game' }
    }));
  }
};

// 处理心跳消息
const handleHeartbeat = (ws, payload, roomManager) => {
  const { roomId } = payload;
  if (roomId) {
    const room = roomManager.getRoom(roomId);
    if (room) {
      // 更新房间最后活动时间
      room.lastActivityTime = Date.now();
      // 可以选择性地回应心跳请求
      ws.send(JSON.stringify({
        type: 'HEARTBEAT_RESPONSE',
        payload: { roomId: roomId, timestamp: Date.now() }
      }));
    }
  }
};

module.exports = { setupWebSocket };