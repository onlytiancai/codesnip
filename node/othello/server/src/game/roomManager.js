const { Room } = require('./room');

class RoomManager {
  constructor() {
    this.rooms = new Map();
    this.roomTimeout = 30 * 60 * 1000; // 30分钟超时
    this.checkInterval = 5 * 60 * 1000; // 每5分钟检查一次
    this.startTimeoutChecker();
  }

  // 创建新房间
  createRoom() {
    const roomId = this.generateRoomId();
    const room = new Room(roomId);
    this.rooms.set(roomId, room);
    return room;
  }

  // 获取房间
  getRoom(roomId) {
    return this.rooms.get(roomId);
  }

  // 删除房间
  deleteRoom(roomId) {
    return this.rooms.delete(roomId);
  }

  // 添加玩家到房间
  addPlayerToRoom(roomId, ws, playerName) {
    const room = this.getRoom(roomId);
    if (!room) {
      return { success: false, message: 'Room not found' };
    }
    
    return room.addPlayer(ws, playerName);
  }

  // 从房间移除玩家
  removePlayerFromRoom(roomId, ws) {
    const room = this.getRoom(roomId);
    if (room) {
      const removedPlayer = room.removePlayer(ws);
      if (removedPlayer) {
        // 广播玩家离开通知给房间内所有玩家
        const message = JSON.stringify({
          type: 'PLAYER_LEFT',
          payload: {
            playerName: removedPlayer.name,
            playerColor: removedPlayer.color
          }
        });
        room.broadcast(message);
      }
      // 如果房间为空，删除房间
      if (room.players.length === 0) {
        this.deleteRoom(roomId);
      }
    }
  }

  // 通过socket移除玩家
  removePlayerBySocket(ws) {
    for (const room of this.rooms.values()) {
      const removedPlayer = room.removePlayer(ws);
      if (removedPlayer) {
        // 广播玩家离开通知给房间内所有玩家
        const message = JSON.stringify({
          type: 'PLAYER_LEFT',
          payload: {
            playerName: removedPlayer.name,
            playerColor: removedPlayer.color
          }
        });
        room.broadcast(message);
        // 如果房间为空，删除房间
        if (room.players.length === 0) {
          this.deleteRoom(room.id);
        }
        return true;
      }
    }
    return false;
  }

  // 玩家落子
  makeMove(roomId, ws, row, col) {
    const room = this.getRoom(roomId);
    if (!room) {
      return { success: false, message: 'Room not found' };
    }
    
    return room.makeMove(ws, row, col);
  }

  // 广播消息给房间内所有玩家
  broadcastToRoom(roomId, message, exclude = []) {
    const room = this.getRoom(roomId);
    if (room) {
      room.broadcast(message, exclude);
    }
  }

  // 生成唯一房间ID
  generateRoomId() {
    let roomId;
    do {
      roomId = Math.random().toString(36).substr(2, 6).toUpperCase();
    } while (this.rooms.has(roomId));
    return roomId;
  }

  // 启动超时检查器
  startTimeoutChecker() {
    setInterval(() => {
      this.checkRoomTimeouts();
    }, this.checkInterval);
  }

  // 检查房间超时
  checkRoomTimeouts() {
    const now = Date.now();
    const roomsToRemove = [];

    for (const [roomId, room] of this.rooms.entries()) {
      if (now - room.getLastActivityTime() > this.roomTimeout) {
        roomsToRemove.push(roomId);
      }
    }

    // 关闭并移除超时的房间
    for (const roomId of roomsToRemove) {
      const room = this.rooms.get(roomId);
      if (room) {
        room.closeRoom();
        this.deleteRoom(roomId);
        console.log(`Room ${roomId} closed due to inactivity`);
      }
    }
  }

  // 关闭房间
  closeRoom(roomId) {
    const room = this.rooms.get(roomId);
    if (room) {
      room.closeRoom();
      this.deleteRoom(roomId);
      return true;
    }
    return false;
  }

  // 获取当前房间数
  getRoomCount() {
    return this.rooms.size;
  }

  // 获取当前在线人数
  getOnlinePlayerCount() {
    let count = 0;
    for (const room of this.rooms.values()) {
      count += room.players.length;
    }
    return count;
  }
}

module.exports = { RoomManager };