# WebSocket 消息协议

## 基本格式
所有 WebSocket 消息都使用 JSON 格式，包含 `type` 和 `payload` 字段：

```json
{
  "type": "消息类型",
  "payload": {
    // 消息内容
  }
}
```

## 客户端到服务器消息

### 1. CREATE_ROOM - 创建房间
请求创建一个新的游戏房间。

**格式：**
```json
{
  "type": "CREATE_ROOM",
  "payload": {}
}
```

**响应：** `ROOM_CREATED`

### 2. JOIN_ROOM - 加入房间
使用房间 ID 加入游戏房间。

**格式：**
```json
{
  "type": "JOIN_ROOM",
  "payload": {
    "roomId": "房间ID",
    "playerName": "玩家名称"
  }
}
```

**响应：** `JOINED_ROOM` 或 `ERROR`

### 3. LEAVE_ROOM - 离开房间
离开当前所在的游戏房间。

**格式：**
```json
{
  "type": "LEAVE_ROOM",
  "payload": {
    "roomId": "房间ID"
  }
}
```

**响应：** 无直接响应，房间内其他玩家会收到 `ROOM_UPDATED`

### 4. MAKE_MOVE - 落子
在指定位置落子。

**格式：**
```json
{
  "type": "MAKE_MOVE",
  "payload": {
    "roomId": "房间ID",
    "row": 行号,
    "col": 列号
  }
}
```

**响应：** `MOVE_MADE` 或 `ERROR`

### 5. GET_ROOM_INFO - 获取房间信息
获取指定房间的详细信息。

**格式：**
```json
{
  "type": "GET_ROOM_INFO",
  "payload": {
    "roomId": "房间ID"
  }
}
```

**响应：** `ROOM_INFO` 或 `ERROR`

## 服务器到客户端消息

### 1. ROOM_CREATED - 房间创建成功
创建房间成功后返回房间 ID。

**格式：**
```json
{
  "type": "ROOM_CREATED",
  "payload": {
    "roomId": "房间ID"
  }
}
```

### 2. JOINED_ROOM - 加入房间成功
成功加入房间后返回房间信息和玩家信息。

**格式：**
```json
{
  "type": "JOINED_ROOM",
  "payload": {
    "roomId": "房间ID",
    "playerId": "玩家ID",
    "playerColor": 玩家颜色 (1=黑, 2=白, null=观战),
    "room": {
      "id": "房间ID",
      "players": [玩家列表],
      "gameState": {
        "board": 棋盘状态,
        "currentPlayer": 当前玩家,
        "scores": { "black": 黑棋数量, "white": 白棋数量 },
        "gameOver": 是否结束,
        "winner": 赢家 (1=黑, 2=白, null=平局),
        "validMoves": 合法落子位置
      }
    }
  }
}
```

### 3. ROOM_UPDATED - 房间状态更新
房间内有玩家加入/离开或游戏状态变化时发送。

**格式：**
```json
{
  "type": "ROOM_UPDATED",
  "payload": {
    "room": {
      "id": "房间ID",
      "players": [玩家列表],
      "gameState": 游戏状态
    }
  }
}
```

### 4. MOVE_MADE - 落子结果
落子操作的结果。

**格式：**
```json
{
  "type": "MOVE_MADE",
  "payload": {
    "success": true/false,
    "message": "操作结果消息",
    "board": 棋盘状态,
    "currentPlayer": 当前玩家,
    "flipped": 翻转的棋子位置,
    "scores": { "black": 黑棋数量, "white": 白棋数量 },
    "gameOver": 是否结束,
    "winner": 赢家
  }
}
```

### 5. ROOM_INFO - 房间信息
返回指定房间的详细信息。

**格式：**
```json
{
  "type": "ROOM_INFO",
  "payload": {
    "room": {
      "id": "房间ID",
      "players": [玩家列表],
      "gameState": 游戏状态
    }
  }
}
```

### 6. ERROR - 错误信息
操作失败时返回的错误信息。

**格式：**
```json
{
  "type": "ERROR",
  "payload": {
    "message": "错误描述"
  }
}
```

## 数据结构定义

### 1. 棋盘状态 (board)
8×8 二维数组，使用数字表示棋子：
- 0: 空
- 1: 黑棋
- 2: 白棋

```json
[
  [0, 0, 0, 0, 0, 0, 0, 0],
  [0, 0, 0, 0, 0, 0, 0, 0],
  // ...
  [0, 0, 1, 2, 0, 0, 0, 0],
  [0, 0, 2, 1, 0, 0, 0, 0],
  // ...
  [0, 0, 0, 0, 0, 0, 0, 0]
]
```

### 2. 玩家 (Player)
```json
{
  "id": "玩家ID",
  "name": "玩家名称",
  "color": 玩家颜色 (1=黑, 2=白, null=观战),
  "isSpectator": 是否为观战者
}
```

### 3. 合法落子位置 (validMoves)
```json
[
  { "row": 2, "col": 3 },
  { "row": 3, "col": 2 },
  { "row": 4, "col": 5 },
  { "row": 5, "col": 4 }
]
```

### 4. 翻转的棋子 (flipped)
```json
[
  { "row": 3, "col": 3 },
  { "row": 4, "col": 4 }
]
```