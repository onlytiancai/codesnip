# WebSocket 消息协议

## 基础格式

所有 WebSocket 消息都使用 JSON 格式，包含以下字段：

```json
{
  "type": "消息类型",
  "payload": { "消息数据" }
}
```

## 消息类型

### 客户端发送消息

#### 1. CREATE_ROOM
- 描述：创建一个新房间
- 格式：
  ```json
  {
    "type": "CREATE_ROOM",
    "payload": {}
  }
  ```

#### 2. JOIN_ROOM
- 描述：加入一个房间
- 格式：
  ```json
  {
    "type": "JOIN_ROOM",
    "payload": {
      "roomId": "房间ID",
      "playerName": "玩家名称"
    }
  }
  ```

#### 3. LEAVE_ROOM
- 描述：离开房间
- 格式：
  ```json
  {
    "type": "LEAVE_ROOM",
    "payload": {
      "roomId": "房间ID"
    }
  }
  ```

#### 4. MAKE_MOVE
- 描述：玩家落子
- 格式：
  ```json
  {
    "type": "MAKE_MOVE",
    "payload": {
      "roomId": "房间ID",
      "row": 3,  // 行索引（0-7）
      "col": 2   // 列索引（0-7）
    }
  }
  ```

#### 5. GET_ROOM_INFO
- 描述：获取房间信息
- 格式：
  ```json
  {
    "type": "GET_ROOM_INFO",
    "payload": {
      "roomId": "房间ID"
    }
  }
  ```

### 服务器发送消息

#### 1. ROOM_CREATED
- 描述：房间创建成功
- 格式：
  ```json
  {
    "type": "ROOM_CREATED",
    "payload": {
      "roomId": "创建的房间ID"
    }
  }
  ```

#### 2. JOINED_ROOM
- 描述：加入房间成功
- 格式：
  ```json
  {
    "type": "JOINED_ROOM",
    "payload": {
      "roomId": "房间ID",
      "playerId": "玩家ID",
      "playerColor": 1,  // 1: 黑棋, 2: 白棋, null: 观战者
      "room": {
        "id": "房间ID",
        "players": [
          {
            "id": "玩家ID",
            "name": "玩家名称",
            "color": 1,  // 1: 黑棋, 2: 白棋, null: 观战者
            "isSpectator": false
          }
        ],
        "gameState": {
          "board": [[0, 0, ...], ...],  // 8x8 棋盘
          "currentPlayer": 1,          // 当前回合玩家
          "scores": { "black": 2, "white": 2 },  // 分数
          "gameOver": false,           // 游戏是否结束
          "winner": null,              // 赢家（null 为平局）
          "validMoves": [{"row": 3, "col": 2}, ...]  // 合法落子位置
        },
        "maxPlayers": 2
      }
    }
  }
  ```

#### 3. ROOM_UPDATED
- 描述：房间状态更新（如玩家加入/离开）
- 格式：同 JOINED_ROOM 中的 room 字段

#### 4. MOVE_MADE
- 描述：落子成功并更新游戏状态
- 格式：
  ```json
  {
    "type": "MOVE_MADE",
    "payload": {
      "success": true,
      "board": [[0, 0, ...], ...],  // 更新后的棋盘
      "currentPlayer": 2,          // 下一个回合玩家
      "flipped": [{"row": 3, "col": 3}, ...],  // 翻转的棋子
      "scores": { "black": 3, "white": 1 },  // 更新后的分数
      "gameOver": false,           // 游戏是否结束
      "winner": null               // 赢家（null 为平局）
    }
  }
  ```

#### 5. ERROR
- 描述：错误消息
- 格式：
  ```json
  {
    "type": "ERROR",
    "payload": {
      "message": "错误描述"
    }
  }
  ```

## 数据结构定义

### 棋盘状态 (Board)
- 8x8 二维数组
- 0: 空格
- 1: 黑棋
- 2: 白棋

### 玩家 (Player)
```javascript
{
  id: "玩家唯一ID",
  name: "玩家名称",
  color: 1 | 2 | null,  // 1: 黑棋, 2: 白棋, null: 观战者
  isSpectator: boolean  // 是否为观战者
}
```

### 房间 (Room)
```javascript
{
  id: "房间唯一ID",
  players: [Player],  // 房间内玩家列表
  gameState: GameState,  // 当前游戏状态
  maxPlayers: 2  // 最大玩家数（不包括观战者）
}
```

### 游戏状态 (GameState)
```javascript
{
  board: Board,  // 棋盘状态
  currentPlayer: 1 | 2,  // 当前回合玩家
  scores: {
    black: number,  // 黑棋数量
    white: number   // 白棋数量
  },
  gameOver: boolean,  // 游戏是否结束
  winner: 1 | 2 | null,  // 赢家（null 为平局）
  validMoves: [{row: number, col: number}]  // 合法落子位置
}
```

### 落子结果 (MoveResult)
```javascript
{
  success: boolean,  // 是否落子成功
  board: Board,  // 更新后的棋盘
  currentPlayer: 1 | 2,  // 下一个回合玩家
  flipped: [{row: number, col: number}],  // 翻转的棋子
  scores: {
    black: number,
    white: number
  },
  gameOver: boolean,
  winner: 1 | 2 | null,
  message?: string  // 错误消息（如果 success 为 false）
}
```