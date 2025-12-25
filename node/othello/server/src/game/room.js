const { OthelloGame } = require('./game');
const { Player } = require('./player');

class Room {
  constructor(id) {
    this.id = id;
    this.players = [];
    this.game = new OthelloGame();
    this.maxPlayers = 2;
  }

  // 添加玩家到房间
  addPlayer(ws, name) {
    // 检查房间是否已满
    const activePlayers = this.players.filter(p => !p.isSpectator);
    if (activePlayers.length >= this.maxPlayers) {
      // 超过最大玩家数，作为观战者加入
      const player = new Player(ws, name);
      this.players.push(player);
      return { success: true, player, room: this.getPublicInfo() };
    }
    
    // 分配颜色
    const usedColors = activePlayers.map(p => p.color);
    const availableColors = [this.game.BLACK, this.game.WHITE].filter(color => !usedColors.includes(color));
    const color = availableColors[0];
    
    const player = new Player(ws, name, color);
    this.players.push(player);
    return { success: true, player, room: this.getPublicInfo() };
  }

  // 移除玩家
  removePlayer(ws) {
    const index = this.players.findIndex(p => p.socket === ws);
    if (index !== -1) {
      this.players.splice(index, 1);
      return true;
    }
    return false;
  }

  // 通过socket获取玩家
  getPlayerBySocket(ws) {
    return this.players.find(p => p.socket === ws);
  }

  // 玩家落子
  makeMove(ws, row, col) {
    const player = this.getPlayerBySocket(ws);
    if (!player || player.isSpectator) {
      return { success: false, message: 'Only active players can make moves' };
    }
    
    if (this.game.currentPlayer !== player.color) {
      return { success: false, message: 'Not your turn' };
    }
    
    return this.game.makeMove(row, col);
  }

  // 获取房间的公共信息
  getPublicInfo() {
    return {
      id: this.id,
      players: this.players.map(p => p.getPublicInfo()),
      gameState: this.game.getGameState(),
      maxPlayers: this.maxPlayers
    };
  }

  // 广播消息给房间内所有玩家
  broadcast(message, exclude = []) {
    this.players.forEach(player => {
      if (!exclude.includes(player.socket) && player.socket.readyState === player.socket.OPEN) {
        player.socket.send(message);
      }
    });
  }
}