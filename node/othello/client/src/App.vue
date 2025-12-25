<template>
  <div class="min-h-screen bg-gray-100 flex flex-col items-center justify-center p-4">
    <h1 class="text-4xl font-bold text-gray-800 mb-8">奥赛罗棋</h1>
    
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
  switch (message.type) {
    case 'ROOM_CREATED':
      roomId.value = message.payload.roomId;
      break;
      
    case 'JOINED_ROOM':
      isInRoom.value = true;
      currentRoomId.value = message.payload.roomId;
      playerColor.value = message.payload.playerColor;
      const room = message.payload.room;
      updateGameState(room.gameState);
      break;
      
    case 'ROOM_UPDATED':
      const updatedRoom = message.payload.room;
      updateGameState(updatedRoom.gameState);
      break;
      
    case 'MOVE_MADE':
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
      
    case 'ERROR':
      alert(message.payload.message);
      break;
  }
};

// 更新游戏状态
const updateGameState = (gameState: any) => {
  board.value = gameState.board;
  currentPlayer.value = gameState.currentPlayer;
  scores.value = gameState.scores;
  gameOver.value = gameState.gameOver;
  winner.value = gameState.winner;
  validMoves.value = gameState.validMoves || [];
};

// 创建房间
const createRoom = () => {
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

// 落子
const makeMove = (row: number, col: number) => {
  if (gameOver.value || playerColor.value !== currentPlayer.value) {
    return;
  }
  
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
});

onBeforeUnmount(() => {
  if (ws.value) {
    ws.value.close();
  }
});
</script>