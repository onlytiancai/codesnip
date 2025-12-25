class OthelloGame {
  constructor() {
    this.BOARD_SIZE = 8;
    this.EMPTY = 0;
    this.BLACK = 1;
    this.WHITE = 2;
    this.currentPlayer = this.BLACK;
    this.board = this.initializeBoard();
    this.gameOver = false;
    this.winner = null;
    this.scores = { black: 2, white: 2 };
  }

  // 初始化棋盘，中间四个棋子
  initializeBoard() {
    const board = Array(this.BOARD_SIZE).fill().map(() => Array(this.BOARD_SIZE).fill(this.EMPTY));
    const mid = Math.floor(this.BOARD_SIZE / 2);
    
    board[mid - 1][mid - 1] = this.WHITE;
    board[mid - 1][mid] = this.BLACK;
    board[mid][mid - 1] = this.BLACK;
    board[mid][mid] = this.WHITE;
    
    return board;
  }

  // 检查落子是否合法
  isValidMove(row, col, player) {
    if (row < 0 || row >= this.BOARD_SIZE || col < 0 || col >= this.BOARD_SIZE) {
      return false;
    }
    
    if (this.board[row][col] !== this.EMPTY) {
      return false;
    }
    
    // 检查8个方向是否有可翻转的棋子
    const directions = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],          [0, 1],
      [1, -1],  [1, 0], [1, 1]
    ];
    
    for (const [dr, dc] of directions) {
      if (this.hasValidDirection(row, col, dr, dc, player)) {
        return true;
      }
    }
    
    return false;
  }

  // 检查特定方向是否有可翻转的棋子
  hasValidDirection(row, col, dr, dc, player) {
    let r = row + dr;
    let c = col + dc;
    let foundOpponent = false;
    
    while (r >= 0 && r < this.BOARD_SIZE && c >= 0 && c < this.BOARD_SIZE) {
      const piece = this.board[r][c];
      
      if (piece === this.EMPTY) {
        return false;
      }
      
      if (piece === player) {
        return foundOpponent;
      }
      
      foundOpponent = true;
      r += dr;
      c += dc;
    }
    
    return false;
  }

  // 落子并翻转棋子
  makeMove(row, col) {
    if (this.gameOver || !this.isValidMove(row, col, this.currentPlayer)) {
      return { success: false, message: 'Invalid move' };
    }
    
    this.board[row][col] = this.currentPlayer;
    const flipped = this.flipPieces(row, col, this.currentPlayer);
    
    // 更新分数
    this.updateScores();
    
    // 检查游戏是否结束
    if (this.isGameOver()) {
      this.gameOver = true;
      this.winner = this.determineWinner();
      return {
        success: true,
        board: this.board,
        currentPlayer: this.currentPlayer,
        flipped,
        scores: this.scores,
        gameOver: true,
        winner: this.winner
      };
    }
    
    // 切换玩家
    const nextPlayer = this.currentPlayer === this.BLACK ? this.WHITE : this.BLACK;
    
    // 检查下一个玩家是否有合法走法
    if (this.hasValidMoves(nextPlayer)) {
      this.currentPlayer = nextPlayer;
    } else {
      // 下一个玩家无子可下，检查当前玩家是否还有合法走法
      if (!this.hasValidMoves(this.currentPlayer)) {
        // 双方都无子可下，游戏结束
        this.gameOver = true;
        this.winner = this.determineWinner();
        return {
          success: true,
          board: this.board,
          currentPlayer: this.currentPlayer,
          flipped,
          scores: this.scores,
          gameOver: true,
          winner: this.winner
        };
      }
      // 只有下一个玩家无子可下，当前玩家继续
    }
    
    return {
      success: true,
      board: this.board,
      currentPlayer: this.currentPlayer,
      flipped,
      scores: this.scores,
      gameOver: false
    };
  }

  // 翻转棋子
  flipPieces(row, col, player) {
    const directions = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1],          [0, 1],
      [1, -1],  [1, 0], [1, 1]
    ];
    const flipped = [];
    
    for (const [dr, dc] of directions) {
      if (this.hasValidDirection(row, col, dr, dc, player)) {
        let r = row + dr;
        let c = col + dc;
        
        while (this.board[r][c] !== player) {
          this.board[r][c] = player;
          flipped.push({ row: r, col: c });
          r += dr;
          c += dc;
        }
      }
    }
    
    return flipped;
  }

  // 检查玩家是否有合法走法
  hasValidMoves(player) {
    for (let row = 0; row < this.BOARD_SIZE; row++) {
      for (let col = 0; col < this.BOARD_SIZE; col++) {
        if (this.isValidMove(row, col, player)) {
          return true;
        }
      }
    }
    return false;
  }

  // 获取所有合法走法
  getValidMoves(player) {
    const moves = [];
    for (let row = 0; row < this.BOARD_SIZE; row++) {
      for (let col = 0; col < this.BOARD_SIZE; col++) {
        if (this.isValidMove(row, col, player)) {
          moves.push({ row, col });
        }
      }
    }
    return moves;
  }

  // 更新分数
  updateScores() {
    let black = 0;
    let white = 0;
    
    for (let row = 0; row < this.BOARD_SIZE; row++) {
      for (let col = 0; col < this.BOARD_SIZE; col++) {
        if (this.board[row][col] === this.BLACK) {
          black++;
        } else if (this.board[row][col] === this.WHITE) {
          white++;
        }
      }
    }
    
    this.scores = { black, white };
  }

  // 检查游戏是否结束
  isGameOver() {
    // 检查棋盘是否满
    const hasEmpty = this.board.some(row => row.some(cell => cell === this.EMPTY));
    if (!hasEmpty) {
      return true;
    }
    
    // 检查双方是否都无子可下
    return !this.hasValidMoves(this.BLACK) && !this.hasValidMoves(this.WHITE);
  }

  // 确定赢家
  determineWinner() {
    const { black, white } = this.scores;
    if (black > white) return this.BLACK;
    if (white > black) return this.WHITE;
    return null; // 平局
  }

  // 获取游戏状态
  getGameState() {
    return {
      board: this.board,
      currentPlayer: this.currentPlayer,
      scores: this.scores,
      gameOver: this.gameOver,
      winner: this.winner,
      validMoves: this.getValidMoves(this.currentPlayer)
    };
  }

  // 重置游戏
  reset() {
    this.board = this.initializeBoard();
    this.currentPlayer = this.BLACK;
    this.gameOver = false;
    this.winner = null;
    this.scores = { black: 2, white: 2 };
  }
}

module.exports = { OthelloGame };