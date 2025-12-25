class Player {
  constructor(ws, name, color = null) {
    this.id = Math.random().toString(36).substr(2, 9);
    this.socket = ws;
    this.name = name;
    this.color = color; // BLACK or WHITE, or null for spectator
    this.isSpectator = color === null;
  }

  // 获取玩家的公共信息（不包含socket）
  getPublicInfo() {
    return {
      id: this.id,
      name: this.name,
      color: this.color,
      isSpectator: this.isSpectator
    };
  }
}

module.exports = { Player };