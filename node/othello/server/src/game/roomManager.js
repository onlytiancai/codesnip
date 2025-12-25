const { Room } = require('./room');

class RoomManager {
  constructor() {
    this.rooms = new Map();
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
      room.removePlayer(ws);
      // 如果房间为空，删除房间
      if (room.players.length === 0) {
        this.deleteRoom(roomId);
      }
    }
  }

  // 通过socket移除玩家
  removePlayerBySocket(ws) {
    for (const room of this.rooms.values()) {
      if (room.removePlayer(ws)) {
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
}

module.exports = { RoomManager };