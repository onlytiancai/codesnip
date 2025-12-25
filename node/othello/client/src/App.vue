<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
    <h1 class="text-4xl font-bold text-gray-800 mb-4">奥赛罗棋</h1>
    
    <!-- 统计信息 -->
    <div v-if="!isInRoom" class="mb-6 text-center">
      <p class="text-gray-600 mb-2">当前房间数: {{ stats.roomCount }}</p>
      <p class="text-gray-600">当前在线人数: {{ stats.onlinePlayerCount }}</p>
    </div>
    
    <!-- 房间创建/加入界面 -->
    <div v-if="!isInRoom" class="bg-white p-8 rounded-lg shadow-lg w-full max-w-md">
      <div class="mb-6">
        <button 
          @click="createRoom" 
          class="w-full bg-blue-500 hover:bg-blue-600 text-white font-bold py-3 px-4 rounded-lg transition duration-300"
        >
          创建新房间
        </button>
      </div>
      
      <div class="border-t border-gray-200 pt-6">
        <h2 class="text-xl font-semibold mb-4">加入房间</h2>
        <div class="flex gap-3">
          <input
            v-model="roomId"
            type="text"
            placeholder="房间ID"
            class="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            @click="joinRoom"
            class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
          >
            加入
          </button>
        </div>
      </div>
    </div>
    
    <!-- 游戏界面 -->
    <div v-else class="w-full max-w-4xl">
      <div class="bg-white p-6 rounded-lg shadow-lg mb-6">
        <div class="flex justify-between items-center mb-4">
          <div>
            <h2 class="text-2xl font-bold">房间ID: {{ currentRoomId }}</h2>
            <p class="text-gray-600">当前回合: {{ currentPlayer === 1 ? '黑棋' : '白棋' }}</p>
            <p class="text-gray-500 text-sm" v-if="!gameOver">游戏状态: {{ roomInfo?.gameStarted ? '已开始' : '等待玩家加入' }}</p>
            <p class="text-gray-500 text-sm" v-if="playerColor">您是: {{ playerColor === 1 ? '黑方' : '白方' }}</p>
            <p class="text-yellow-500 text-sm" v-if="!gameOver && currentPlayer !== playerColor">
              等待对方落子: {{ waitTime }}秒
            </p>
          </div>
          <div class="flex gap-6">
            <div class="text-center">
              <p class="font-semibold">黑棋</p>
              <p class="text-2xl font-bold text-gray-800">{{ scores.black }}</p>
            </div>
            <div class="text-center">
              <p class="font-semibold">白棋</p>
              <p class="text-2xl font-bold text-gray-800">{{ scores.white }}</p>
            </div>
            <div>
              <button 
                @click="leaveRoom" 
                class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-4 rounded-lg transition duration-300"
              >
                退出房间
              </button>
            </div>
          </div>
        </div>
        
        <!-- 棋盘组件 -->
        <Board 
          :board="board" 
          :currentPlayer="currentPlayer" 
          :validMoves="validMoves" 
          :playerColor="playerColor"
          :gameOver="gameOver"
          @makeMove="makeMove"
        />
        
        <!-- 游戏结束信息 -->
        <div v-if="gameOver" class="mt-6 text-center">
          <h3 class="text-2xl font-bold mb-2">游戏结束</h3>
          <p class="text-xl" :class="winner === playerColor ? 'text-green-600' : winner ? 'text-red-600' : 'text-gray-600'">
            {{ winner ? (winner === playerColor ? '你赢了！' : '你输了！') : '平局！' }}
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount } from 'vue';
import Board from './components/Board.vue';

// 游戏状态
const isInRoom = ref(false);
const roomId = ref('');
const currentRoomId = ref('');
const playerColor = ref<number | null>(null);
const board = ref<number[][]>([]);
const currentPlayer = ref(1);
const scores = ref({ black: 2, white: 2 });
const gameOver = ref(false);
const winner = ref<number | null>(null);
const validMoves = ref<{ row: number; col: number }[]>([]);
const roomInfo = ref<any>(null);

// 统计信息
const stats = ref({ roomCount: 0, onlinePlayerCount: 0 });

// 计时器
const waitTime = ref(0);
let waitTimer: number | null = null;

// WebSocket连接
const ws = ref<WebSocket | null>(null);
const wsUrl = 'ws://localhost:3001';

// 创建WebSocket连接
const connectWebSocket = () => {
  ws.value = new WebSocket(wsUrl);
  
  ws.value.onopen = () => {
    console.log('WebSocket connected');
  };
  
  ws.value.onmessage = (event) => {
    const message = JSON.parse(event.data);
    handleWebSocketMessage(message);
  };
  
  ws.value.onclose = () => {
    console.log('WebSocket disconnected');
    // 尝试重连
    setTimeout(connectWebSocket, 3000);
  };
  
  ws.value.onerror = (error) => {
    console.error('WebSocket error:', error);
  };
};

// 处理WebSocket消息
const handleWebSocketMessage = (message: any) => {
  console.log('Received WebSocket message:', message);
  switch (message.type) {
    case 'ROOM_CREATED':
      console.log('Room created:', message.payload.roomId);
      roomId.value = message.payload.roomId;
      break;
      
    case 'JOINED_ROOM':
      console.log('Joined room:', message.payload.roomId);
      isInRoom.value = true;
      currentRoomId.value = message.payload.roomId;
      playerColor.value = message.payload.playerColor;
      const room = message.payload.room;
      roomInfo.value = room;
      updateGameState(room.gameState);
      break;
      
    case 'ROOM_UPDATED':
      console.log('Room updated:', message.payload.room.id);
      const updatedRoom = message.payload.room;
      roomInfo.value = updatedRoom;
      updateGameState(updatedRoom.gameState);
      break;
      
    case 'MOVE_MADE':
      console.log('Move made:', message.payload);
      if (message.payload.success) {
        updateGameState({
          board: message.payload.board,
          currentPlayer: message.payload.currentPlayer,
          scores: message.payload.scores,
          gameOver: message.payload.gameOver,
          winner: message.payload.winner,
          validMoves: [] // 会在ROOM_UPDATED中获取
        });
      }
      break;
      
    case 'ROOM_CLOSED':
      console.log('Room closed:', message.payload.message);
      alert(message.payload.message);
      isInRoom.value = false;
      currentRoomId.value = '';
      playerColor.value = null;
      stopWaitTimer();
      break;
      
    case 'STATS':
      console.log('Received stats:', message.payload);
      stats.value = message.payload;
      break;
      
    case 'ERROR':
      console.error('WebSocket error:', message.payload.message);
      alert(message.payload.message);
      break;
  }
};

// 更新游戏状态
const updateGameState = (gameState: any) => {
  board.value = gameState.board;
  const previousPlayer = currentPlayer.value;
  currentPlayer.value = gameState.currentPlayer;
  scores.value = gameState.scores;
  gameOver.value = gameState.gameOver;
  winner.value = gameState.winner;
  validMoves.value = gameState.validMoves || [];

  // 控制计时器
  if (gameOver.value) {
    stopWaitTimer();
  } else if (playerColor.value && currentPlayer.value !== playerColor.value) {
    // 对方回合，启动计时器
    startWaitTimer();
  } else {
    // 自己回合，停止计时器
    stopWaitTimer();
  }
};

// 启动等待计时器
const startWaitTimer = () => {
  stopWaitTimer();
  waitTime.value = 0;
  waitTimer = window.setInterval(() => {
    waitTime.value++;
  }, 1000);
};

// 停止等待计时器
const stopWaitTimer = () => {
  if (waitTimer) {
    clearInterval(waitTimer);
    waitTimer = null;
  }
  waitTime.value = 0;
};

// 创建房间
const createRoom = () => {
  console.log('Creating new room');
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'CREATE_ROOM',
      payload: {}
    }));
  }
};

// 加入房间
const joinRoom = () => {
  if (!roomId.value.trim()) {
    alert('请输入房间ID');
    return;
  }
  
  const playerName = prompt('请输入您的名字:');
  if (!playerName) return;
  
  console.log('Joining room:', roomId.value, 'as', playerName);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'JOIN_ROOM',
      payload: {
        roomId: roomId.value,
        playerName: playerName
      }
    }));
  }
};

// 退出房间
const leaveRoom = () => {
  console.log('Leaving room:', currentRoomId.value);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'LEAVE_ROOM',
      payload: {
        roomId: currentRoomId.value
      }
    }));
  }
  // 停止计时器
  stopWaitTimer();
  // 重置游戏状态
  isInRoom.value = false;
  currentRoomId.value = '';
  playerColor.value = null;
  roomInfo.value = null;
  // 重置游戏板和分数
  board.value = [];
  scores.value = { black: 2, white: 2 };
  gameOver.value = false;
  winner.value = null;
  validMoves.value = [];
};

// 落子
const makeMove = (row: number, col: number) => {
  if (gameOver.value || playerColor.value !== currentPlayer.value) {
    return;
  }
  
  console.log('Making move at:', row, col);
  if (ws.value && ws.value.readyState === WebSocket.OPEN) {
    ws.value.send(JSON.stringify({
      type: 'MAKE_MOVE',
      payload: {
        roomId: currentRoomId.value,
        row: row,
        col: col
      }
    }));
  }
};

// 生命周期钩子
onMounted(() => {
  connectWebSocket();
  // 每30秒更新一次统计信息
  const statsInterval = setInterval(() => {
    if (ws.value && ws.value.readyState === WebSocket.OPEN && !isInRoom.value) {
      ws.value.send(JSON.stringify({ type: 'GET_STATS', payload: {} }));
    }
  }, 30000);

  onBeforeUnmount(() => {
    clearInterval(statsInterval);
  });
});

onBeforeUnmount(() => {
  if (ws.value) {
    ws.value.close();
  }
  stopWaitTimer();
});
</script>