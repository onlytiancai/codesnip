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
    default:
      ws.send(JSON.stringify({
        type: 'ERROR',
        payload: { message: 'Unknown message type' }
      }));
  }
};

const handleCreateRoom = (ws, payload, roomManager) => {
  const room = roomManager.createRoom();
  ws.send(JSON.stringify({
    type: 'ROOM_CREATED',
    payload: { roomId: room.id }
  }));
};

const handleJoinRoom = (ws, payload, roomManager) => {
  const { roomId, playerName } = payload;
  const result = roomManager.addPlayerToRoom(roomId, ws, playerName);
  
  if (result.success) {
    ws.send(JSON.stringify({
      type: 'JOINED_ROOM',
      payload: {
        roomId: roomId,
        playerId: result.player.id,
        playerColor: result.player.color,
        room: result.room
      }
    }));
    // 广播房间更新给其他玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'ROOM_UPDATED',
      payload: { room: result.room }
    }), [ws]);
  } else {
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
  const result = roomManager.makeMove(roomId, ws, row, col);
  
  if (result.success) {
    // 广播移动结果给所有玩家
    roomManager.broadcastToRoom(roomId, JSON.stringify({
      type: 'MOVE_MADE',
      payload: result
    }));
  } else {
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

module.exports = { setupWebSocket };